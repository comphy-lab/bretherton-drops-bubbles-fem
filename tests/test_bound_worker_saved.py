"""Synthetic parent-envelope controls; no saved-paper numerics or native execution."""

from __future__ import annotations

import copy
import importlib.abc
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from runners import run_bound_worker as worker


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import in software-only wrapper tests")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


FAKE_CHILD = r'''"""Tiny software child: no numerical libraries, solver or factor."""
import argparse, hashlib, json, sys
from pathlib import Path
SCENARIO = __SCENARIO__
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(path): return {"sha256":sha(path),"bytes":path.stat().st_size}
def write(path, value): path.write_text(json.dumps(value, indent=2)+"\n")
p=argparse.ArgumentParser()
p.add_argument("--packet",type=Path,required=True)
p.add_argument("--packet-sha256",required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--check",action="store_true")
p.add_argument("--execute",action="store_true")
p.add_argument("--shift",required=True)
a=p.parse_args()
assert sys.flags.isolated == 1 and a.shift == "0.001"
assert a.check != a.execute and sha(a.packet) == a.packet_sha256
packet=json.loads(a.packet.read_text());base=a.packet.parent;t=packet["target"]
witness={}
for name,item in packet["inputs"].items():
    path=Path(item["path"]);path=path if path.is_absolute() else base/path
    witness[name]={"path":str(path),**record(path)}
    assert witness[name]["sha256"]==item["sha256"]
if a.check:
    checked={"status":"metadata_checked_no_execution","packet_sha256":a.packet_sha256,
        "source_closure":packet["source_closure"],"input_witness":witness,"output_absent":True,
        "native_imports":False,"factorisations":0,"eigensolves":0,
        "n":t["n"],"J_nnz":t["j_nnz"],"M_nnz":t["m_nnz"],
        "mass_rows":t["mass_rows"],"mass_columns":t["mass_columns"],
        "globals_preserved":t["globals"],"forced":t["forced"],
        "model_hash":witness["case"]["sha256"],"time_factor":t["time_factor"],
        "pencil_sign":packet["authority"]["pencil_sign"]}
    if SCENARIO=="bad_metadata": checked["factorisations"]=1
    print(json.dumps(checked));raise SystemExit(0)
assert not a.output.exists()
a.output.mkdir()
if SCENARIO=="nonzero":
    write(a.output/"failure.json",{"error_message":"ORIGINAL_SOFTWARE_CHILD_ERROR"})
    print("ORIGINAL_SOFTWARE_CHILD_STDOUT")
    print("ORIGINAL_SOFTWARE_CHILD_ERROR",file=sys.stderr)
    raise SystemExit(7)
status="budget_or_requested_stop" if SCENARIO=="budget" else "no_additional_challenger_found_among_numerically_classified_checked_roots" if SCENARIO=="negative" else "computed_additional_candidate_passes_saved_pencil_gates"
result={"status":status,"post_input_hashes_verified":True,"no_native_import":True,
    "second_shift_automatic":False,"physical_mode_qualification":False,
    "finite_spectrum_complete":False,"global_dominance_certified":False,"software_only":True}
write(a.output/"result.json",result)
first={"event":"prepared","packet_sha256":a.packet_sha256}
final={"event":"final","status":status,"result_sha256":sha(a.output/"result.json")}
if SCENARIO=="bad_progress": final["result_sha256"]="0"*64
(a.output/"progress.jsonl").write_text(json.dumps(first)+"\n"+json.dumps(final)+"\n")
if SCENARIO=="missing_completion": raise SystemExit(0)
files={path.name:record(path)for path in a.output.iterdir()}
write(a.output/"output-hashes.json",files)
files["output-hashes.json"]=record(a.output/"output-hashes.json")
write(a.output/"completion.json",{"schema":"stone-paper-saved-spectrum-output-v1","status":status,
    "files":files,"scientific_acceptance":False,"finite_spectrum_complete":False,
    "global_dominance_certified":False,"native_imports":False})
if SCENARIO=="tamper": (a.output/"result.json").write_text("{}")
if SCENARIO=="input_change":
    path=Path(packet["inputs"]["j"]["path"])
    (path if path.is_absolute() else base/path).write_text("changed software input")
print(json.dumps({"completion":str(a.output/"completion.json"),"completion_sha256":sha(a.output/"completion.json"),"exit_code":0}))
'''


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from problems import paper_bubble as model
    from runners import run_paper_case as runner

    root = tmp_path / "inputs"
    packet_root = root / "packet"
    packet_root.mkdir(parents=True)
    output = tmp_path / "runtime"
    output.mkdir()
    original = model.read_case(
        worker.ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    )
    original["case_id"] = "software_saved_anchor"
    original["physics"]["ca_in"] = -0.00315
    original_path = packet_root / "evidence/case.json"
    write_json(original_path, original)
    inputs = {}
    for name in worker.INPUT_ROLES:
        if name == "case":
            path = original_path
        else:
            path = packet_root / "evidence" / (name + ".txt")
            path.write_text("software-only input " + name)
        inputs[name] = {
            "path": str(path.relative_to(packet_root)),
            "sha256": worker.digest(path),
            "bytes": path.stat().st_size,
            "kind": "json" if name == "case" else "text",
        }
    environment = {
        "available": True,
        "commit": worker.PIN,
        "direct_url": {
            "url": "https://github.com/comphy-lab/pyoomph",
            "vcs_info": {"vcs": "git", "commit_id": worker.PIN},
        },
        "url": "https://github.com/comphy-lab/pyoomph",
        "native_core": {"path": "/software/core.so", "sha256": worker.CORE},
        "interpreter": {"resolved_executable": str(Path(sys.executable).resolve())},
    }
    component = {"commit": "1" * 40, "head": "1" * 40, "clean": True, "status": [], "sources": {}}
    monkeypatch.setattr(runner, "component_revision", lambda root: copy.deepcopy(component))
    monkeypatch.setattr(runner, "pyoomph_revision", lambda: copy.deepcopy(environment))
    monkeypatch.setattr(runner, "_git", lambda *args: Path(worker.__file__).read_bytes())
    config_path, case_path = root / "configuration.json", root / "dispatch-case.json"
    packet_path = packet_root / "packet.json"
    packet = {
        "schema": worker.PACKET_SCHEMA,
        "authority": {
            "source_commit": worker.PRODUCER,
            "pyoomph_commit": worker.PIN,
            "native_core_sha256": worker.CORE,
            "pencil_sign": "J=-dR/dU; Jv=lambda Mv",
        },
        "environment": {
            "interpreter": str(worker.CANONICAL_PYTHON),
            "resolved_interpreter": str(worker.CANONICAL_PYTHON.resolve()),
        },
        "inputs": inputs,
        "source_closure": {},
        "model": original,
        "target": {
            "n": 316989,
            "j_nnz": 10488602,
            "m_nnz": 11836,
            "mass_rows": 1481,
            "mass_columns": 2960,
            "globals": {"globals/U": [316987], "gas_volume/pressure": [316988]},
            "forced": [],
            "time_factor": 3.3 ** (-1 / 3),
        },
        "gates": {},
        "resource_caps": {},
        "algorithm": {"shift": 0.001},
        "reference_rectangles": [],
        "protected_roots": [],
    }
    config = {
        "case": str(case_path),
        "entrypoint": "packet/saved_search.py",
        "files": {},
        "worker_interface": worker.SAVED_INTERFACE,
        "packet": {},
    }
    b = SimpleNamespace(
        root=root,
        packet_root=packet_root,
        output=output,
        model=original,
        original_path=original_path,
        config_path=config_path,
        case_path=case_path,
        packet_path=packet_path,
        packet=packet,
        config=config,
        component=component,
        environment=environment,
        runner=runner,
    )

    def publish(scenario="success"):
        for name in worker.CLOSURE:
            path = packet_root / name
            if name == "saved_search.py":
                path.write_text(FAKE_CHILD.replace("__SCENARIO__", repr(scenario)))
            elif not path.exists():
                path.write_text('"""Software-only unused closure witness."""\n')
            packet["source_closure"][name] = {
                "sha256": worker.digest(path),
                "bytes": path.stat().st_size,
            }
        write_json(packet_path, packet)
        config["packet"] = {"path": "packet/packet.json", "sha256": worker.digest(packet_path)}
        config["files"] = {
            "packet/" + name: item["sha256"] for name, item in packet["source_closure"].items()
        }
        write_json(config_path, config)
        b.case = {
            **copy.deepcopy(original),
            "configuration": {"path": str(config_path), "sha256": worker.digest(config_path)},
        }
        write_json(case_path, b.case)

    def argv(preflight=False):
        value = [
            "--case",
            str(case_path),
            "--configuration",
            str(config_path),
            "--output",
            str(output),
        ]
        return value + ["--preflight-only"] if preflight else value

    b.publish, b.argv = publish, argv
    publish()
    return b


@pytest.fixture
def bound_component(bundle, tmp_path, monkeypatch):
    """Use real method bytes in private copies; never alter the component under test."""
    original_root = worker.ROOT
    root = tmp_path / "component"
    methods = set(bundle.runner.METHOD_FILES) | {
        str(path.relative_to(original_root)) for path in (original_root / "problems").glob("*.py")
    }
    for name in methods | {"runners/run_bound_worker.py"}:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((original_root / name).read_bytes())
    bundle.component["sources"] = {
        name: {
            "sha256": worker.digest(root / name),
            "committed_sha256": worker.digest(root / name),
            "committed_matches": True,
        }
        for name in methods
    }
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "__file__", str(root / "runners/run_bound_worker.py"))
    monkeypatch.setattr(bundle.runner, "__file__", str(root / "runners/run_paper_case.py"))
    monkeypatch.setattr(
        sys.modules["problems.paper_bubble"], "__file__", str(root / "problems/paper_bubble.py")
    )
    bundle.source_root = root
    bundle.original_source_root = original_root
    return bundle


def no_source_command_after_prepared(bundle, argv, kwargs):
    if (bundle.output / "progress.jsonl").exists():
        assert Path(argv[0]).name != "git", "Git subprocess after prepared"
        assert Path(kwargs.get("cwd", Path.cwd())).resolve() not in {
            bundle.source_root,
            bundle.original_source_root,
        }


@pytest.mark.parametrize("scenario", ["success", "budget", "negative"])
def test_no_source_subprocess_after_prepared_through_parent_exit(
    bound_component, monkeypatch, scenario
):
    b = bound_component
    b.publish(scenario)
    monkeypatch.chdir(b.output.parent)
    original_revision, original_git = b.runner.component_revision, b.runner._git
    original_run, original_popen = worker.subprocess.run, worker.subprocess.Popen
    original_record = worker.file_record
    source_phases, execution_calls, post_child_sources, post_child_authorities = (
        [],
        [],
        set(),
        set(),
    )
    child_finished = False

    def source_call(function, *args):
        assert not (b.output / "progress.jsonl").exists(), "source subprocess after prepared"
        assert not (b.output / "run-manifest.json").exists(), "Git rebind after parent manifest"
        source_phases.append(function)
        return function(*args)

    monkeypatch.setattr(
        b.runner, "component_revision", lambda *args: source_call(original_revision, *args)
    )
    monkeypatch.setattr(b.runner, "_git", lambda *args: source_call(original_git, *args))

    def popen(argv, *args, **kwargs):
        no_source_command_after_prepared(b, argv, kwargs)
        return original_popen(argv, *args, **kwargs)

    def run(argv, **kwargs):
        nonlocal child_finished
        no_source_command_after_prepared(b, argv, kwargs)
        if (b.output / "progress.jsonl").exists():
            execution_calls.append(argv)
        result = original_run(argv, **kwargs)
        if "--execute" in argv:
            child_finished = True
        return result

    def record(path, *args, **kwargs):
        if child_finished and Path(path).is_relative_to(b.source_root):
            post_child_sources.add(Path(path))
        if child_finished and Path(path).is_relative_to(b.root):
            post_child_authorities.add(Path(path))
        return original_record(path, *args, **kwargs)

    monkeypatch.setattr(worker.subprocess, "run", run)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    monkeypatch.setattr(worker, "file_record", record)
    assert worker.main(b.argv()) == 0
    assert source_phases and len(execution_calls) == 1
    assert "--execute" in execution_calls[0] and execution_calls[0][1] == "-I"
    assert post_child_sources == {
        *(b.source_root / name for name in b.component["sources"]),
        Path(worker.__file__),
    }
    assert post_child_authorities == {
        b.case_path,
        b.config_path,
        b.packet_path,
        *(b.packet_root / name for name in worker.CLOSURE),
        *(b.packet_root / item["path"] for item in b.packet["inputs"].values()),
    }
    assert (b.output / "completion.json").is_file()
    assert all(not name.startswith(("pyoomph", "_pyoomph")) for name in sys.modules)


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "wrapper",
        "core_input",
        "environment_core",
        "environment_metadata",
        "interpreter",
        "module_custody",
    ],
)
def test_post_child_byte_environment_and_module_changes_are_refused(
    bound_component, monkeypatch, fault
):
    b = bound_component
    original_run = worker.subprocess.run

    def changed(argv, **kwargs):
        result = original_run(argv, **kwargs)
        if "--execute" in argv:
            if fault in {"method", "wrapper"}:
                path = b.source_root / (
                    "problems/forcing.py" if fault == "method" else "runners/run_bound_worker.py"
                )
                path.write_bytes(path.read_bytes() + b"\n# changed software-only method\n")
            elif fault == "core_input":
                (b.packet_root / b.packet["inputs"]["core"]["path"]).write_text(
                    "changed software core"
                )
            elif fault == "environment_core":
                b.environment["native_core"]["sha256"] = "0" * 64
            elif fault == "environment_metadata":
                b.environment["version"] = "changed metadata"
            elif fault == "interpreter":
                b.environment["interpreter"]["resolved_executable"] = "/different/interpreter"
            else:
                monkeypatch.setattr(
                    sys.modules["problems.paper_bubble"],
                    "__file__",
                    str(b.source_root / "elsewhere.py"),
                )
        return result

    monkeypatch.setattr(worker.subprocess, "run", changed)
    assert worker.main(b.argv()) != 0
    assert not (b.output / "completion.json").exists()
    failure = json.loads((b.output / "failure.json").read_text())
    assert failure["administrative_complete"] is False
    assert (b.output / "result/completion.json").is_file()


@pytest.mark.parametrize("preflight", [False, True])
def test_legacy_forwarding_is_exactly_unchanged(tmp_path, monkeypatch, preflight):
    case = tmp_path / "case.json"
    case.write_text("{}")
    entrypoint = tmp_path / "legacy.py"
    entrypoint.write_text("# software-only legacy entrypoint\n")
    config = tmp_path / "configuration.json"
    write_json(
        config,
        {
            "case": str(case),
            "entrypoint": entrypoint.name,
            "files": {entrypoint.name: worker.digest(entrypoint)},
            "old_extra": True,
        },
    )
    output = tmp_path / "legacy-output"
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(worker.subprocess, "run", run)
    args = ["--case", str(case), "--configuration", str(config), "--output", str(output)]
    if preflight:
        args.append("--preflight-only")
    assert worker.main(args) == 17
    assert calls == [
        (
            [
                sys.executable,
                str(entrypoint),
                "--case",
                str(case),
                "--configuration",
                str(config),
                "--output",
                str(output),
            ]
            + (["--preflight-only"] if preflight else []),
            {"check": False},
        )
    ]
    assert not output.exists()


def test_saved_metadata_only_creates_no_parent_or_child_output(bundle, capsys):
    b = bundle
    assert worker.main(b.argv(preflight=True)) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["parent_output_written"] is False and checked["native_imports"] is False
    assert not list(b.output.iterdir())


@pytest.mark.parametrize("scenario", ["success", "budget", "negative"])
def test_sealed_child_outcome_is_administrative_only(bundle, scenario):
    b = bundle
    b.publish(scenario)
    assert worker.main(b.argv()) == 0
    parent = json.loads((b.output / "completion.json").read_text())
    summary = json.loads((b.output / "summary.json").read_text())
    result = json.loads((b.output / "result.json").read_text())
    assert parent["status"] == "complete" and summary["completion_reached"] is True
    assert (
        result["child_status"]
        == json.loads((b.output / "result/result.json").read_text())["status"]
    )
    assert all(parent[name] is False for name in worker.NO_CLAIMS)
    assert parent["excluded_mutable_administrative_files"] == ["monitor.jsonl"]
    for name, expected in parent["files"].items():
        assert worker.digest(b.output / name) == expected
    final = json.loads((b.output / "progress.jsonl").read_text().splitlines()[-1])
    assert final["event"] == "final" and final["status"] == "completed"
    manifest = json.loads((b.output / "run-manifest.json").read_text())
    assert manifest["component_commit"] != manifest["saved_producer_commit"]
    assert manifest["child_argv"][1] == "-I"
    assert manifest["child_argv"][-3:] == ["--execute", "--shift", "0.001"]
    assert "--stop-after-block" not in manifest["child_argv"]
    assert manifest["case_sha256"] == worker.digest(b.case_path)
    assert manifest["runner"]["sha256"] == worker.digest(Path(worker.__file__))


@pytest.mark.parametrize(
    "scenario", ["nonzero", "missing_completion", "tamper", "bad_progress", "input_change"]
)
def test_child_failure_is_preserved_and_never_completed(bundle, scenario):
    b = bundle
    b.publish(scenario)
    assert worker.main(b.argv()) != 0
    assert not (b.output / "completion.json").exists()
    failure = json.loads((b.output / "failure.json").read_text())
    assert failure["administrative_complete"] is False and failure["numerical_acceptance"] is False
    if scenario == "nonzero":
        assert failure["child_returncode"] == 7
        assert "ORIGINAL_SOFTWARE_CHILD_ERROR" in failure["child_stderr"]
        assert (
            json.loads((b.output / "result/failure.json").read_text())["error_message"]
            == "ORIGINAL_SOFTWARE_CHILD_ERROR"
        )


@pytest.mark.parametrize(
    "fault",
    [
        "configuration_hash",
        "model",
        "original_case",
        "packet_hash",
        "closure_hash",
        "entrypoint",
        "configuration_extra",
        "missing_input",
        "source",
        "interpreter",
    ],
)
def test_typed_bindings_refuse_mismatches_before_output(bundle, fault, monkeypatch):
    b = bundle
    if fault == "configuration_hash":
        b.case["configuration"]["sha256"] = "0" * 64
        write_json(b.case_path, b.case)
    elif fault == "model":
        b.case["physics"]["ca_in"] = -0.0032
        write_json(b.case_path, b.case)
    elif fault == "original_case":
        original = copy.deepcopy(b.model)
        original["case_id"] = "different_software_model"
        write_json(b.original_path, original)
        item = b.packet["inputs"]["case"]
        item.update(sha256=worker.digest(b.original_path), bytes=b.original_path.stat().st_size)
        b.publish()
    elif fault == "packet_hash":
        b.config["packet"]["sha256"] = "0" * 64
        write_json(b.config_path, b.config)
        b.case["configuration"]["sha256"] = worker.digest(b.config_path)
        write_json(b.case_path, b.case)
    elif fault == "closure_hash":
        (b.packet_root / "numerics.py").write_text("changed software closure")
    elif fault == "entrypoint":
        b.config["entrypoint"] = "packet/numerics.py"
        b.publish()
    elif fault == "configuration_extra":
        b.config["flags"] = ["--stop-after-block", "1"]
        b.publish()
    elif fault == "missing_input":
        b.packet["inputs"].pop("j")
        b.publish()
    elif fault == "source":
        monkeypatch.setattr(b.runner, "_git", lambda *args: b"changed wrapper commit bytes")
    else:
        b.packet["environment"]["interpreter"] = "/usr/bin/python3"
        b.publish()
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.main(b.argv(preflight=True))
    assert not list(b.output.iterdir())


def test_nonempty_runtime_and_symlink_authorities_are_refused(bundle):
    b = bundle
    (b.output / "keep.txt").write_text("user-owned fixture")
    with pytest.raises(ValueError, match="empty"):
        worker.main(b.argv())
    assert (b.output / "keep.txt").read_text() == "user-owned fixture"
    (b.output / "keep.txt").unlink()
    alias = b.root / "configuration-alias.json"
    alias.symlink_to(b.config_path)
    args = b.argv(preflight=True)
    args[3] = str(alias)
    with pytest.raises(ValueError, match="symlink"):
        worker.main(args)
    assert not list(b.output.iterdir())


def test_cached_native_is_refused_before_helpers(bundle, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyoomph", ModuleType("pyoomph"))
    monkeypatch.setattr(
        worker,
        "metadata_helpers",
        lambda *args: pytest.fail("helpers imported before native refusal"),
    )
    with pytest.raises(ValueError, match="native-import invariant"):
        worker.main(bundle.argv(preflight=True))
    assert not list(bundle.output.iterdir())


def test_numerics_in_metadata_check_are_refused(bundle):
    bundle.publish("bad_metadata")
    with pytest.raises(ValueError, match="executed numerics"):
        worker.main(bundle.argv(preflight=True))
    assert not list(bundle.output.iterdir())


def test_administrative_monitor_is_the_only_excluded_parent_file(bundle, monkeypatch):
    b = bundle
    original_seal = worker.seal_parent

    def monitored(prepared, child_files):
        (b.output / "monitor.jsonl").write_text('{"software_only":true}\n')
        return original_seal(prepared, child_files)

    monkeypatch.setattr(worker, "seal_parent", monitored)
    assert worker.main(b.argv()) == 0
    completion = json.loads((b.output / "completion.json").read_text())
    assert "monitor.jsonl" not in completion["files"]
    assert all(not name.endswith("/monitor.jsonl") for name in completion["files"])


@pytest.mark.parametrize(
    "fault",
    ["packet_escape", "closure_symlink", "directory_input", "unknown_interface", "duplicate_case"],
)
def test_regular_contained_typed_authority_controls(bundle, fault):
    b = bundle
    if fault == "packet_escape":
        outside = b.root.parent / "outside-packet.json"
        outside.write_bytes(b.packet_path.read_bytes())
        b.config["packet"] = {"path": str(outside), "sha256": worker.digest(outside)}
        write_json(b.config_path, b.config)
        b.case["configuration"]["sha256"] = worker.digest(b.config_path)
        write_json(b.case_path, b.case)
    elif fault == "closure_symlink":
        path = b.packet_root / "numerics.py"
        raw = path.read_bytes()
        path.unlink()
        target = b.root.parent / "software-numerics.py"
        target.write_bytes(raw)
        path.symlink_to(target)
    elif fault == "directory_input":
        item = b.packet["inputs"]["j"]
        path = b.packet_root / item["path"]
        path.unlink()
        path.mkdir()
    elif fault == "unknown_interface":
        b.config["worker_interface"] = "arbitrary-command-v1"
        write_json(b.config_path, b.config)
    else:
        text = b.case_path.read_text()
        b.case_path.write_text(text.replace("{\n", '{\n  "case_id": "duplicate",\n', 1))
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.main(b.argv(preflight=True))
    assert not list(b.output.iterdir())


def test_final_progress_precedes_seal_and_child_bytes_are_unchanged(bundle, monkeypatch):
    b = bundle
    original_check = worker.child_completion
    original_write = b.runner._exclusive_json
    original_progress = b.runner.append_progress
    snapshot, ordering = {}, []

    def checked(prepared):
        value = original_check(prepared)
        snapshot.update({name: item["sha256"] for name, item in value[1].items()})
        return value

    def written(path, data):
        if path.name == "completion.json":
            assert ordering[-1] == "final"
            assert {
                path.name: worker.digest(path) for path in (b.output / "result").iterdir()
            } == snapshot
            ordering.append("seal")
        return original_write(path, data)

    def progress(output, event, **kwargs):
        ordering.append(event["event"])
        return original_progress(output, event, **kwargs)

    monkeypatch.setattr(worker, "child_completion", checked)
    monkeypatch.setattr(b.runner, "_exclusive_json", written)
    monkeypatch.setattr(b.runner, "append_progress", progress)
    assert worker.main(b.argv()) == 0
    assert ordering == ["prepared", "final", "seal"]
    assert {path.name: worker.digest(path) for path in (b.output / "result").iterdir()} == snapshot


@pytest.mark.parametrize(
    "fault", ["parent_file", "parent_directory", "child_file", "child_history"]
)
def test_sealing_refuses_unexpected_or_changed_artifacts(bundle, monkeypatch, fault):
    b = bundle
    original = worker.seal_parent

    def changed(prepared, child_files):
        if fault == "parent_file":
            (b.output / "unexpected.txt").write_text("software-only unexpected artifact")
        elif fault == "parent_directory":
            (b.output / "unexpected").mkdir()
        elif fault == "child_file":
            (b.output / "result/unexpected.txt").write_text("software-only unexpected artifact")
        else:
            with (b.output / "result/progress.jsonl").open("a") as stream:
                stream.write('{"event":"unexpected"}\n')
        return original(prepared, child_files)

    monkeypatch.setattr(worker, "seal_parent", changed)
    assert worker.main(b.argv()) != 0
    assert not (b.output / "completion.json").exists()
    assert json.loads((b.output / "failure.json").read_text())["status"] == "failed"


def test_helper_native_import_attempt_is_denied(bundle, monkeypatch):
    monkeypatch.setattr(bundle.runner, "pyoomph_revision", lambda: __import__("pyoomph"))
    with pytest.raises(ImportError, match="native solver import denied"):
        worker.main(bundle.argv(preflight=True))
    assert not list(bundle.output.iterdir())
