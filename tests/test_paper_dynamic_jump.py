"""Software contracts only; all state, fold and flow fixtures are synthetic."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.abc
import json
import math
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph.") or "_pyoomph" in fullname:
            raise AssertionError("native import is forbidden in this software suite")


_import_guard = NoNative()
sys.meta_path.insert(0, _import_guard)
try:
    from runners import paper_dynamic_jump as study
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import run_paper_dynamic_jump as cli
finally:
    sys.meta_path.remove(_import_guard)


@pytest.fixture(autouse=True)
def deny_native_imports():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    try:
        yield
    finally:
        sys.meta_path.remove(guard)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
TEMPLATE = ROOT / "validation/herradaEtAl_paper_model/studies/four-role-jumps.template.json"


def full_witness(ca, *, time=8.0, counter=4, dt=0.25, steady=True):
    data = []
    for index, kind in enumerate(("nodal", "position", "internal", "ode")):
        data.append({"name": kind + "/data0", "kind": kind,
                     "stepper_index": 0,
                     "fields": [kind + "/field0"], "pins": [False], "equations": [index],
                     "rows": [[index + level + 0.125] for level in range(7)]})
    return {"schema": "stone-jump-complete-physical-v3", "ndof": 4, "data": data,
            "census": {"schema": "stone-raw-data-census-v1", "counts": {kind: 1 for kind in ("nodal", "position", "internal", "ode")},
                "occurrences": [{"site": row["name"], "kind": row["kind"], "owner": row["name"]} for row in data],
                "external": [], "enumerated_meshes": ["synthetic"],
                "ownership": {"origin": "synthetic", "pyoomph_pin": study.PYOOMPH_PIN,
                              "pointer_observed": False, "registered_count": 3}},
            "policy": {"all_parameters": {"Ca_in": ca, "Bo": 1.56}, "default_scheme": "BDF2",
                "serialization": {"output_step": counter, "continue_step": 0,
                    "suggested_next_dt": dt, "run_endtime": None, "remesher_counter": 0}},
            "geometry": {"nodes": [[0.0, 1.0], [0.25, 1.25]], "connectivity": [0, 1],
                         "lagrangian": [[0.0, 1.0], [0.25, 1.25]]},
            "layout": {"ndof": 4, "types": [0, 1, 2, 3],
                       "dof_names": [row["fields"][0] for row in data]},
            "parameters": {"Ca_in": ca, "Bo": 1.56}, "time": time, "dts": [dt, dt],
            "counter": counter, "steppers": [{"type": "synthetic.BDF2", "order": 2,
                                               "ntstorage": 7, "is_steady": steady} for _ in range(3)],
            "flags": dict(zip(study.FLAGS, (False, True, True), strict=True)),
            "bc_cache": "stationary", "mode": copy.deepcopy(runtime.BASE_MODE),
            "pending_quality_remesh": {"schema": "stone-jump-owned-quality-remesh-v1",
                "owner": "shared_primary_template", "domains": ["gas", "liquid"], "requested": False}}


def synthetic_dump(mesh, witness):
    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    return (string("pyoomph_dump") + string("0.1.6") + string("global")
            + struct.pack("<dqqqq", witness["time"], witness["policy"]["serialization"]["output_step"], 0, 6, 1)
            + string(mesh) + integer(1) + integer(0) + integer(len(witness["dts"]))
            + b"".join(struct.pack("<d", dt) for dt in witness["dts"])
            + integer(witness["counter"]) + struct.pack("<dd", witness["dts"][0], float("nan"))
            + b"synthetic opaque payload; not native evidence"
            + study.stationary._bc_cache_trailer(witness["bc_cache"]))


class Binder:
    def __init__(self, directory):
        self.directory = directory

    def bind(self, name, payload):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = payload if isinstance(payload, bytes) else (json.dumps(payload, allow_nan=False) + "\n").encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def portable(self, prefix, witness):
        mesh = self.bind(prefix + "/mesh.msh", b"synthetic mesh, never loaded natively")
        state = self.bind(prefix + "/state.dump", synthetic_dump("mesh.msh", witness))
        inventory = self.bind(prefix + "/mesh-inventory.json", {
            "state": state, "mesh_reference_base": str(Path(state["path"]).parent),
            "meshes": [{"stored_path": "mesh.msh", **mesh}]})
        physical = self.bind(prefix + "/physical.json", {"synthetic": True, "native": False,
                                                          "state": state, "witness": witness})
        return state, inventory, physical


def fixture(directory):
    binder = Binder(directory)
    packet = json.loads(TEMPLATE.read_text())
    packet["thresholds"] = {name: 1e-6 for name in study.THRESHOLDS}
    packet["threshold_evidence"] = binder.bind("thresholds.json", {
        "synthetic": True, "native": False, "thresholds": packet["thresholds"],
        "preregistered_before_trajectory": True,
        "length_estimator": "existing_curved_interface_meridional_length_over_R",
        "profile_frame": "front_tip", "norms": {name: "synthetic declared norm" for name in study.PROFILE_NORMS}})
    controls = packet["controls"]
    controls["max_newton_iterations_per_step"], controls["max_total_newton_updates"] = 4, 1000
    controls["remesh"].update(max_per_run=2, max_inversion_retries_per_step=1)
    controls["startup_registration_receipt"] = binder.bind("startup.json", {
        "synthetic": True, "native": False, "startup": study.STARTUP,
        "stepper_activation": study.ACTIVATION, "temporal_scale": 1.0,
        "preregistered_before_trajectory": True})
    core = binder.bind("synthetic-core.bin", b"synthetic core; not an installed extension")
    environment = {"available": True, "commit": study.PYOOMPH_PIN, "url": "https://github.com/comphy-lab/pyoomph",
                   "direct_url": {"vcs_info": {"vcs": "git", "commit_id": study.PYOOMPH_PIN}}, "native_core": core}
    source = {"head": study.FEM_BASE, "commit": study.FEM_BASE, "clean": True,
              "status": [], "sources": {}, "declared_commit": None}
    proof = binder.bind("synthetic-proof.json", {"synthetic": True, "native": False})
    for direction in packet["directions"]:
        initial, target = direction.split("_to_")
        ca = -0.005 if initial == "I" else -0.004
        lo, hi = ca - 0.0001, ca + 0.0001
        width = max(ca - lo, hi - ca)
        target_ca = ca + (2 * width if initial == "I" else -2 * width)
        case = json.loads(CASE.read_text())
        case["case_id"], case["physics"]["ca_in"] = "synthetic_fold_" + initial, ca
        case_ref = binder.bind(initial + "/case.json", case)
        raw = binder.bind(initial + "/fold-result.json", {"synthetic": True, "native": False,
            "status": "complete", "ca_fold": ca, "fold_case": case,
            **{name: True for name in study.FOLD_GATES - {"branch_identity_qualified"}}})
        witness = full_witness(ca)
        state, inventory, physical = binder.portable(initial, witness)
        manifest = binder.bind(initial + "/run-manifest.json", {
            "synthetic": True, "native": False, "case_sha256": case_ref["sha256"],
            "case_id": case["case_id"], "component": source, "component_commit": source["commit"],
            "pyoomph": environment})
        restart = binder.bind(initial + "/resume.json", {
            "schema": study.stationary.RESUME_SCHEMA, "case_sha256": case_ref["sha256"],
            "parent_manifest": manifest, "state": {**state, "format": "portable_file"},
            "mesh_reference_base": str(Path(state["path"]).parent),
            "meshes": json.loads(Path(inventory["path"]).read_text())["meshes"],
            "azimuthal_stability": True,
            "stage": {"kind": "completed_polish", "bond_index": 0, "bond": 1.56,
                      "ca_in": ca, "bc_mode": "stationary"}, "roundtrip": {"verified": True}})
        uncertainty = {"origin": "measured_total_fold_uncertainty", "ca_interval": [lo, hi],
            "radius": width, "physical_and_numerical_sources_bound": True,
            "method_and_error_budget": "synthetic total envelope; no numerical claim",
            "sources": {name: {"included": True, "justification": "synthetic fixture", "evidence": [proof]}
                        for name in study.UNCERTAINTY_SOURCES}}
        uncertainty_ref = binder.bind(initial + "/uncertainty.json", uncertainty)
        cert = binder.bind(initial + "/parent.json", {"schema": "stone-jump-parent-v1", "status": "complete",
            "synthetic": True, "native": False, "branch": initial, "ca_fold": ca,
            "model": study.MODEL, "pyoomph_pin": study.PYOOMPH_PIN,
            "checks": {name: True for name in study.FOLD_GATES}, "uncertainty": uncertainty,
            "artifacts": {"case": case_ref, "manifest": manifest, "fold_result": raw,
                          "restart_packet": restart, "state": state, "mesh_inventory": inventory,
                          "physical_signature": physical, "uncertainty_evidence": uncertainty_ref},
            "branch_connection_evidence": binder.bind(initial + "/branch.json", {
                "synthetic": True, "native": False, "branch": initial, "qualified": True,
                "fold_result": raw, "case": case_ref, "model": study.MODEL}),
            "native_load_and_next_step_qualification": binder.bind(initial + "/next-step.json", {
                "synthetic": True, "native": False, "state": state, "physical_signature": physical,
                "case": case_ref, "native_core": core, "pyoomph_pin": study.PYOOMPH_PIN,
                "full_history_load_passed": True, "unaugmented_base_mode_passed": True,
                "transient_next_step_passed": True})})
        end_case = study.target_case(case, target_ca, direction)
        end_state, end_inventory, end_physical = binder.portable(direction + "-endpoint", full_witness(target_ca))
        end_case_ref = binder.bind(direction + "-endpoint/case.json", end_case)
        end_qualification = binder.bind(direction + "-endpoint/qualification.json", {
            "synthetic": True, "native": False, "status": "complete", "numerical_acceptance": True,
            "branch": target, "ca_in": target_ca, "state": end_state,
            "physical_signature": end_physical, "stationary_residual": 0.0})
        end = binder.bind(direction + "-endpoint/certificate.json", {"schema": "stone-jump-endpoint-v1",
            "synthetic": True, "native": False, "branch": target, "ca_in": target_ca, "qualified": True,
            "model": study.MODEL, "pyoomph_pin": study.PYOOMPH_PIN,
            "artifacts": {"case": end_case_ref, "state": end_state, "mesh_inventory": end_inventory,
                          "physical_signature": end_physical, "qualification": end_qualification}})
        horizon = 1250 if initial == "I" else 3050
        packet["directions"][direction].update(parent_certificate=cert, endpoint_certificate=end,
            target_ca=target_ca, horizon_hat=horizon, final_window_hat=50,
            endpoint_qualification={"fresh_cold_load_required": True, "max_stationary_calls": 0,
                                    "max_newton_updates": 0, "max_remeshes": 0},
            offset={"origin": "measured_fold_uncertainty", "preregistered_before_trajectory": True,
                    "factor": 2.0, "maximum_delta": 0.001, "registration_receipt": binder.bind(direction + "/offset.json", {
                        "synthetic": True, "native": False, "direction": direction, "target_ca": target_ca,
                        "factor": 2.0, "maximum_delta": 0.001, "parent_certificate": cert,
                        "preregistered_before_trajectory": True})})
    for role in packet["roles"]:
        role["max_steps"] = packet["directions"][role["direction"]]["horizon_hat"] // role["dt_hat"]
    return packet, binder


@pytest.fixture
def bundle(tmp_path):
    packet, binder = fixture(tmp_path)
    return SimpleNamespace(packet=packet, binder=binder)


def test_four_roles_native_mapping_and_two_sided_profiles_have_no_authority(bundle):
    plan = study.build_plan(bundle.packet)
    assert plan["ready_for_dispatch"] is False and plan["native_execution_authorized"] is False
    assert plan["native_imported"] is False
    assert [row["max_steps"] for row in plan["roles"]] == [25, 50, 61, 122]
    assert [row["dt_solver"] for row in plan["roles"]] == [study.RATIO * 50, study.RATIO * 25] * 2
    support = plan["roles"][0]["time_support"]
    assert [(row["lower_t_hat"], row["upper_t_hat"]) for row in support] == [(0, 50), (200, 250), (400, 450), (1000, 1000)]
    assert all(row["interpolated"] is False for role in plan["roles"] for row in role["time_support"])
    assert plan["native_verification_gates"]


@pytest.mark.parametrize("change", [
    lambda p: p["model"].update(gravity="upward"),
    lambda p: p["model"].update(gas_viscosity_ratio=0),
    lambda p: p["model"].update(r0=True),
    lambda p: p["source"].update(pyoomph_pin="0" * 40),
    lambda p: p["controls"].update(temporal_scale=study.RATIO),
    lambda p: p["controls"].update(temporal_scale=True),
    lambda p: p["controls"].pop("temporal_error"),
    lambda p: p["controls"].update(startup="pseudo_time_replay"),
    lambda p: p["controls"].update(allow_profile_interpolation=True),
    lambda p: p["controls"]["remesh"].update(automatic_postsolve_quality_remesh=True),
    lambda p: p["controls"]["remesh"].update(max_per_run=None),
    lambda p: p["thresholds"].update(remesh_history_profile_norm=None),
    lambda p: p["directions"]["I_to_III"].update(horizon_hat=1210),
    lambda p: p["directions"]["I_to_III"].update(horizon_hat=float("inf")),
    lambda p: p["directions"]["I_to_III"].update(target_ca=0.05),
    lambda p: p["directions"]["III_to_I"]["offset"].update(factor=1),
    lambda p: p["roles"][1].update(dt_hat=12.5),
    lambda p: p["roles"][1].update(dt_solver=25),
    lambda p: p["roles"][0].update(max_steps=26),
    lambda p: p["roles"][0].update(output_every_steps=2),
    lambda p: p["execution"].update(native_execution_authorized=True),
])
def test_unbound_control_model_role_and_budget_drift_refuses(bundle, change):
    change(bundle.packet)
    with pytest.raises(ValueError):
        study.build_plan(bundle.packet)


def test_parent_replacement_false_direct_flags_and_foreign_evidence_refuse(bundle):
    reference = bundle.packet["directions"]["I_to_III"]["parent_certificate"]
    Path(reference["path"]).write_text("{}")
    with pytest.raises(ValueError, match="digest changed"):
        study.build_plan(bundle.packet)
    packet, binder = fixture(bundle.binder.directory)
    reference = packet["directions"]["I_to_III"]["parent_certificate"]
    cert = json.loads(Path(reference["path"]).read_text())
    raw = json.loads(Path(cert["artifacts"]["fold_result"]["path"]).read_text())
    raw["cold_load_verified"] = False
    cert["artifacts"]["fold_result"] = binder.bind("I/fold-result.json", raw)
    packet["directions"]["I_to_III"]["parent_certificate"] = binder.bind("I/parent.json", cert)
    with pytest.raises(ValueError, match="direct fold result is unqualified"):
        study.build_plan(packet)
    packet, binder = fixture(bundle.binder.directory)
    reference = packet["directions"]["I_to_III"]["parent_certificate"]
    cert = json.loads(Path(reference["path"]).read_text())
    cert["branch_connection_evidence"] = binder.bind("I/branch.json", {"qualified": True, "branch": "III"})
    packet["directions"]["I_to_III"]["parent_certificate"] = binder.bind("I/parent.json", cert)
    with pytest.raises(ValueError, match="branch connection"):
        study.build_plan(packet)


def test_regular_authority_and_unknown_norms_are_required(bundle):
    link = bundle.binder.directory / "symlink.json"
    link.symlink_to(Path(bundle.packet["threshold_evidence"]["path"]))
    bundle.packet["threshold_evidence"]["path"] = str(link)
    with pytest.raises(ValueError, match="regular authority"):
        study.build_plan(bundle.packet)
    packet, binder = fixture(bundle.binder.directory)
    evidence = json.loads(Path(packet["threshold_evidence"]["path"]).read_text())
    evidence["norms"]["pair_profile_norm"] = None
    packet["threshold_evidence"] = binder.bind("thresholds.json", evidence)
    with pytest.raises(ValueError, match="definition is unbound"):
        study.build_plan(packet)


def test_cli_real_template_and_native_modes_refuse_before_import_or_output(bundle, tmp_path):
    filled = Path(bundle.binder.bind("study.json", bundle.packet)["path"])
    for mode, packet_path in (("--check", TEMPLATE), ("--check", filled), ("--start-jump", filled)):
        arguments = [str(ROOT / "runners/run_paper_dynamic_jump.py"), str(packet_path), "--source", str(ROOT),
                     "--role", "I_to_III_dt_hat_50", mode]
        program = """
import importlib.abc, runpy, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.') or '_pyoomph' in fullname:
            raise AssertionError('native import during denied command')
sys.meta_path.insert(0, BlockNative())
sys.path.insert(0, ROOT_PATH)
from runners import run_paper_case
run_paper_case.pyoomph_revision = lambda: {'available': False, 'commit': None}
sys.argv = ARGS
runpy.run_path(sys.argv[0], run_name='__main__')
""".replace("ARGS", repr(arguments)).replace("ROOT_PATH", repr(str(ROOT)))
        result = subprocess.run([sys.executable, "-B", "-c", program], cwd=tmp_path,
                                capture_output=True, text=True, check=False)
        assert result.returncode == (0 if mode == "--check" and packet_path == filled else 2), result.stderr
        payload = json.loads(result.stdout)
        assert payload["ready_for_dispatch"] is False and payload["native_imported"] is False
        if packet_path == TEMPLATE:
            assert "bounded integer" in payload["refusal"]
        elif mode == "--check":
            assert payload["status"] == "checked" and payload["source_seals_pending"]
        else:
            assert payload["refusal"] == "production native adapter is unqualified"
            assert payload["native_verification_gates"]
    assert not (tmp_path / "solver").exists()


def test_source_closure_freezes_baseline_and_refuses_extra_modules(bundle, tmp_path, monkeypatch):
    checkout = tmp_path / "synthetic-checkout"
    checkout.mkdir()
    frozen = checkout / "problems/frozen.py"
    frozen.parent.mkdir()
    frozen.write_text("FROZEN = 1\n")
    for name in cli.IMPLEMENTATION:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(study.stationary, "component_revision", lambda root: {
        "head": study.FEM_BASE, "commit": study.FEM_BASE, "clean": False})

    def objects(root, *arguments):
        if arguments[0] == "merge-base":
            return b""
        if arguments[0] == "ls-tree":
            return b"problems/frozen.py\n"
        if arguments == ("show", study.FEM_BASE + ":problems/frozen.py"):
            return b"FROZEN = 1\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(study.stationary, "_git", objects)
    _, closure, pending = cli.source_closure(checkout, bundle.packet, study.Artifacts())
    assert set(closure["files"]) == {"problems/frozen.py", *cli.IMPLEMENTATION}
    assert "clean committed isolated source pending" in pending
    frozen.write_text("FROZEN = 2\n")
    with pytest.raises(ValueError, match="frozen physical model/helper changed"):
        cli.source_closure(checkout, bundle.packet, study.Artifacts())
    frozen.write_text("FROZEN = 1\n")
    (checkout / "runners/unsealed.py").write_text("EXTRA = 1\n")
    with pytest.raises(ValueError, match="unsealed additional"):
        cli.source_closure(checkout, bundle.packet, study.Artifacts())


def dynamic_packet(bundle):
    packet, binder = bundle.packet, bundle.binder
    plan = study.build_plan(packet)
    role = plan["roles"][0]
    study_ref = binder.bind("study.json", packet)
    rows = []
    for index in (0, 1):
        witness = full_witness(role["target_ca"], time=index * role["dt_solver"], counter=index,
                               dt=role["dt_solver"], steady=False)
        witness["flags"], witness["bc_cache"] = runtime.DYNAMIC_FLAGS.copy(), "transient"
        state, inventory, physical = binder.portable("dynamic-step-" + str(index), witness)
        p = FakeProblem(plan["bindings"][role["direction"]]["target_case"], witness)
        metrics = fake_hooks(p).diagnostics(p)
        rows.append({"accepted_step_index": index, "t_hat": index * role["dt_hat"],
                     "next_output_index": index,
                     "native_time": index * role["dt_solver"], "counter": index,
                     "dt_solver": role["dt_solver"], "state": state, "mesh_inventory": inventory,
                     "physical_witness": physical, "metrics": metrics,
                     "profile": binder.bind("dynamic-step-" + str(index) + "/profile.json", {"synthetic": True}),
                     "field_qualification": "stationary_geometry_prejump_fluid_fields" if index == 0
                                            else "converged_native_transient"})
    work = runtime.empty_work()
    work.update(newton_updates=1, solve_calls=1, inversion_retries=[0], newton_step_updates=[1])
    result = {"schema": "stone-figure7-dynamic-resume-v3", "study": study_ref,
              "next_output_index": 1,
              "role_id": role["role_id"], "parent_certificate": packet["directions"][role["direction"]]["parent_certificate"],
              "target_case": plan["bindings"][role["direction"]]["target_case"], "dt_hat": role["dt_hat"],
              "dt_solver": role["dt_solver"], "horizon_hat": role["horizon_hat"], "max_steps": role["max_steps"],
              "accepted_step_index": 1, "state": state, "mesh_inventory": inventory, "physical_witness": physical,
              "native_metadata": {"time": role["dt_solver"], "dts": witness["dts"], "unsteady_steps": 1,
                                  "suggested_next_dt": role["dt_solver"], "output_step": 1},
              "steppers": witness["steppers"], "solve_flags": runtime.DYNAMIC_FLAGS.copy(), "work": work,
              "step_ledger": binder.bind("dynamic-steps.json", {"study": study_ref, "role_id": role["role_id"], "rows": rows}),
              "pyoomph_pin": study.PYOOMPH_PIN, "source": packet["source"],
              "native_core": plan["bindings"][role["direction"]]["parent"]["manifest"]["pyoomph"]["native_core"]}
    return result, study_ref, plan, role


def test_native_free_dynamic_packet_binds_raw_time_dt_counter_and_artifacts(bundle):
    resumed, study_ref, plan, role = dynamic_packet(bundle)
    reference = bundle.binder.bind("dynamic-resume.json", resumed)
    checked = cli.validate_dynamic_resume(reference, study_ref, bundle.packet, plan, role, study.Artifacts())
    assert checked["witness"]["counter"] == 1 and checked["work"]["newton_updates"] == 1
    assert checked["mesh_exchange"] == {"mesh.msh": str(Path(resumed["state"]["path"]).parent / "mesh.msh")}
    Path(resumed["state"]["path"]).write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="digest changed"):
        cli.validate_dynamic_resume(reference, study_ref, bundle.packet, plan, role, study.Artifacts())


@pytest.mark.parametrize("change", [
    lambda p: p.update(role_id="III_to_I_dt_hat_50"),
    lambda p: p.update(max_steps=26),
    lambda p: p.update(horizon_hat=1300),
    lambda p: p.update(accepted_step_index=0),
    lambda p: p["native_metadata"].update(time=0),
    lambda p: p["native_metadata"].update(unsteady_steps=0),
    lambda p: p["solve_flags"].update(_taken_already_an_unsteady_step=False),
    lambda p: p["work"].update(newton_updates=1001),
    lambda p: p["work"].update(solve_calls=0),
    lambda p: p["work"].update(inversion_retries=[2]),
    lambda p: p.update(native_core={"path": "/different/core", "sha256": "0" * 64}),
])
def test_dynamic_packet_drift_cannot_reset_startup_or_extend_work(bundle, change):
    resumed, study_ref, plan, role = dynamic_packet(bundle)
    change(resumed)
    reference = bundle.binder.bind("dynamic-resume.json", resumed)
    with pytest.raises(ValueError):
        cli.validate_dynamic_resume(reference, study_ref, bundle.packet, plan, role, study.Artifacts())


class Clock:
    def __init__(self, problem):
        self.problem = problem

    def dt(self, index):
        return self.problem.witness["dts"][index]

    def ndt(self):
        return len(self.problem.witness["dts"])


class Stepper:
    def __init__(self, problem, index, steady):
        self.problem, self.index, self.steady = problem, index, steady

    def is_steady(self):
        return self.steady

    def make_steady(self):
        self.steady = True

    def undo_make_steady(self):
        self.problem.calls.append("undo_steady_" + str(self.index))
        assert all(dt > 0 for dt in self.problem.witness["dts"])
        self.steady = False
        if self.problem.fault == "activation_history":
            self.problem.witness["data"][0]["rows"][1][0] += 1

    def time_pt(self):
        return Clock(self.problem)

    def get_num_unsteady_steps_done(self):
        return self.problem.witness["counter"]

    def set_num_unsteady_steps_done(self, count):
        self.problem.calls.append("reset_counter")
        self.problem.witness["counter"] = count


class Template:
    def __init__(self, problem):
        self.problem, self._template_override = problem, None
        self.remesher = None

    def _get_template(self):
        return self if self._template_override is None else self._template_override

    def get_problem(self):
        return self.problem

    def has_domain(self, name):
        return name in {"liquid", "gas"}


class Remesher:
    def __init__(self, template):
        self.template, self._cnt = template, 0


class FakeProblem:
    def __init__(self, case, saved):
        self.case, self.saved, self.witness = copy.deepcopy(case), copy.deepcopy(saved), copy.deepcopy(saved)
        self.calls, self.records, self.rows = [], [], []
        self.fault, self.budget = None, None
        self._last_bc_setting = saved["bc_cache"]
        self._suggested_next_dt = saved["policy"]["serialization"]["suggested_next_dt"]
        self.default_timestepping_scheme = "BDF2"
        self.steppers = [Stepper(self, i, row["is_steady"]) for i, row in enumerate(saved["steppers"])]
        self.timestepper = self.steppers[0]
        for name, value in saved["flags"].items():
            setattr(self, name, value)
        self._bifurcation_reactivation_after_adaptation = None
        self._azimuthal_mode_param_m = SimpleNamespace(value=0)
        self.do_call_remeshing_when_necessary = False
        self.inversion_remesh_max_retries = 1
        original, active = Template(self), Template(self)
        original._template_override = active
        owner = Remesher(active)
        original.remesher = active.remesher = owner
        self._meshtemplate_list = [original]
        self._domains_remesh_on_inversion = {original, active}
        self._domains_to_remesh = set()
        self._meshdict = {name: SimpleNamespace(_templatemesh=active) for name in ("liquid", "gas")}
        self.remesh_requested = False
        self.volume_factor = 1.0

    def raw(self):
        result = copy.deepcopy(self.witness)
        result["flags"] = {name: getattr(self, name) for name in study.FLAGS}
        result["bc_cache"] = self._last_bc_setting
        result["policy"]["all_parameters"] = copy.deepcopy(result["parameters"])
        result["policy"]["serialization"]["suggested_next_dt"] = self._suggested_next_dt
        result["pending_quality_remesh"]["requested"] = bool(self._domains_to_remesh)
        for row, stepper in zip(result["steppers"], self.steppers, strict=True):
            row["is_steady"] = stepper.steady
        return result

    def ntime_stepper(self):
        return len(self.steppers)

    def time_stepper_pt(self, index):
        return self.steppers[index]

    def get_current_time(self, **kwargs):
        return self.witness["time"]

    def get_global_parameter(self, name):
        p = self

        class Parameter:
            @property
            def value(self):
                return p.witness["parameters"][name]

            @value.setter
            def value(self, value):
                p.calls.append("set_" + name)
                p.witness["parameters"][name] = value
        return Parameter()

    def get_global_parameter_names(self):
        return sorted(self.witness["parameters"])

    def get_bifurcation_tracking_mode(self):
        return "fold" if self.fault == "tracker" else ""

    def get_custom_assembler(self):
        return object() if self.fault == "assembler" else None

    def _get_solved_residual(self):
        return ""

    def get_scaling(self, name):
        return study.RATIO if self.fault == "scale" else 1.0

    def load_state(self, state, **kwargs):
        self.calls.append("load_state")
        assert kwargs == {"quiet": True, "additional_info": {"exchange_msh_file": {"mesh.msh": "/synthetic/mesh.msh"}}}
        self.witness = copy.deepcopy(self.saved)
        self._last_bc_setting = self.saved["bc_cache"]
        self._suggested_next_dt = self.saved["policy"]["serialization"]["suggested_next_dt"]
        for name in study.FLAGS:
            setattr(self, name, False)
        for stepper in self.steppers:
            stepper.steady = True
        if self.fault == "load_history":
            self.witness["data"][2]["rows"][2][0] += 1

    def set_current_time(self, value, **kwargs):
        self.calls.append("set_time")
        self.witness["time"] = value

    def assign_initial_values_impulsive(self, dt):
        self.calls.append("impulsive_fill")
        self.witness["dts"] = [dt, dt]
        for block in self.witness["data"]:
            current = block["rows"][0][:]
            block["rows"] = [current[:], current[:], current[:], [0.0], [0.0], [0.0], current[:]]
        if self.fault == "fill_current":
            self.witness["data"][0]["rows"][0][0] += 1

    def actions_before_transient_solve(self, **kwargs):
        self.calls.append("transient_hook")
        self._last_bc_setting = "transient"
        if self.fault == "bc_pins":
            self.witness["data"][0]["pins"][0] = True

    def actions_after_change_in_global_parameter(self, name):
        self.calls.append("parameter_hook")
        if self.fault == "adoption_history":
            self.witness["data"][3]["rows"][2][0] += 1

    def invalidate_eigendata(self):
        self.calls.append("invalidate_eigendata")

    def invalidate_cached_mesh_data(self):
        self.calls.append("invalidate_mesh_cache")

    def solve(self, **kwargs):
        self.calls.append("solve")
        assert kwargs["temporal_error"] is None and kwargs["spatial_adapt"] == 0
        assert kwargs["shift_values"] is True and kwargs["do_not_set_IC"] is True
        assert self.do_call_remeshing_when_necessary is False
        self.budget.before_solve()
        for _ in range(2 if self.fault == "newton_budget" else 1):
            self.budget.before_update()
            self.budget.after_update()
        self.budget.after_solve()
        dt = kwargs["timestep"]
        self.witness["time"] += dt / 2 if self.fault == "clock_clipping" else dt
        self.witness["dts"] = [dt, dt]
        self.witness["counter"] += 1
        for block in self.witness["data"]:
            block["rows"][2] = block["rows"][1][:]
            block["rows"][1] = block["rows"][0][:]
            block["rows"][0][0] += 0.125
        for name, value in runtime.DYNAMIC_FLAGS.items():
            setattr(self, name, value)
        if self.fault == "counter_reset":
            self.witness["counter"] = 0
        if self.fault == "postsolve_remesh":
            self.actions_before_remeshing([self._meshtemplate_list[0].remesher])
        if self.remesh_requested:
            self._domains_to_remesh = {self._meshtemplate_list[0]._get_template()}

    def output(self):
        self.calls.append("output")
        self.witness["policy"]["serialization"]["output_step"] += 1
        if self.fault == "output_history":
            self.witness["data"][0]["rows"][1][0] += 1

    def remeshing_necessary(self):
        return bool(self._domains_to_remesh)

    def actions_before_remeshing(self, active_remeshers):
        pass

    def actions_after_remeshing(self):
        pass

    def remesh_if_necessary(self):
        self.calls.append("remesh")
        self.actions_before_remeshing([self._meshtemplate_list[0].remesher])
        if self.fault == "flatten_history":
            for block in self.witness["data"]:
                block["rows"][1] = block["rows"][2] = block["rows"][0][:]
        if self.fault == "volume_transfer":
            self.volume_factor += 5e-7
        self._domains_to_remesh = set()
        self.actions_after_remeshing()
        return True


def fake_hooks(problem, binder=None):
    @contextlib.contextmanager
    def monitor(p, budget):
        p.budget = budget
        try:
            yield
        finally:
            p.budget = None

    def diagnostic(p):
        expected = math.pi * study.RATIO**2 * p.witness["parameters"]["Ca_in"]
        return {"gas_volume": 4 * math.pi / 3 * p.volume_factor,
                "surface_volume": 4 * math.pi / 3 * p.volume_factor,
                "inverted_elements": False, "interface_conformity": True,
                "meridional_L_over_R": 2.0, "Ca_b": -0.003, "volume_multiplier": 0.5,
                "gas_pressure": 0.5, "transient_newton_residual": 0.0,
                "upper_lab_axial_flux": expected + (0.1 if p.fault == "old_forcing" else 0.0),
                "lower_lab_axial_flux": expected, "lower_static_pressure_residual": 0.0}

    def transfer(before, after):
        # Independent comparison for a synthetic unchanged mesh; no native claim.
        errors = [abs(a - b) for old, new in zip(before["data"], after["data"], strict=True)
                  for ra, rb in zip(old["rows"], new["rows"], strict=True)
                  for a, b in zip(ra, rb, strict=True)]
        return {"mapped_fields": sorted({f for b in before["data"] for f in b["fields"]}),
                "mapped_history_slots": {f: list(range(len(b["rows"]))) for b in before["data"] for f in b["fields"]},
                "topology_unchanged": True, "interface_conformity": True,
                "geometry_norm": 0.0, "history_norm": max(errors)}

    def retain(p, row, witness):
        p.rows.append(copy.deepcopy(row))
        assert binder is not None, "software retention needs an outside-context fixture owner"
        prefix = "retained-" + str(id(p)) + "-" + str(row["accepted_step_index"])
        state, inventory, physical = binder.portable(prefix, witness)
        return {**copy.deepcopy(row), "state": state, "mesh_inventory": inventory,
                "physical_witness": physical,
                "profile": binder.bind(prefix + "/profile.json", {"synthetic": True,
                    "native": False, "geometry": witness["geometry"]})}

    return runtime.RuntimeHooks(capture=lambda p: p.raw(), load_session=lambda p: contextlib.nullcontext(),
                                update_session=monitor, diagnostics=diagnostic, transfer_errors=transfer,
                                record=lambda label, data: problem.records.append((label, copy.deepcopy(data))),
                                retain=retain, template_type=Template, remesher_type=Remesher)


def prepare_fake(bundle):
    plan = study.build_plan(bundle.packet)
    role = copy.deepcopy(plan["roles"][0])
    role["max_steps"], role["horizon_hat"] = 3, 150  # A unit-scale software sequence, not Figure 7.
    binding = plan["bindings"][role["direction"]]
    parent = {**binding["parent"], "mesh_exchange": {"mesh.msh": "/synthetic/mesh.msh"}}
    problem = FakeProblem(parent["case"], parent["witness"])
    hooks = fake_hooks(problem, bundle.binder)
    return problem, hooks, parent, binding["target_case"], role


def test_fresh_sequence_loads_parent_and_fills_without_capsule_or_history_loss(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    current = [block["rows"][0] for block in parent["witness"]["data"]]
    runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    assert p.calls == ["load_state", "set_time", "impulsive_fill", "undo_steady_0", "undo_steady_1",
                       "undo_steady_2", "reset_counter", "transient_hook", "set_Ca_in", "parameter_hook",
                       "invalidate_eigendata", "invalidate_mesh_cache"]
    assert [block["rows"][0] for block in p.witness["data"]] == current
    assert p.witness["counter"] == 0 and p.witness["time"] == 0
    assert p._suggested_next_dt == role["dt_solver"]
    assert p.case == target and p.get_global_parameter("Ca_in").value == role["target_ca"]
    got = runtime.fixed_steps(p, role, bundle.packet["controls"], bundle.packet["thresholds"], hooks,
                              datetime(2099, 1, 1, tzinfo=UTC))
    assert len(got["rows"]) == 4 and [r["t_hat"] for r in got["rows"]] == [0, 50, 100, 150]
    assert got["work"]["newton_updates"] == got["work"]["solve_calls"] == 3
    assert p.calls.count("solve") == 3 and p.calls.count("output") == 3 and "remesh" not in p.calls
    assert got["endpoint_qualified"] is False and got["numerical_acceptance"] is False


@pytest.mark.parametrize("fault", ["load_history", "fill_current", "activation_history", "bc_pins",
                                    "adoption_history", "tracker", "assembler", "scale"])
def test_fresh_guards_refuse_actual_hidden_history_pin_mode_changes(bundle, fault):
    p, hooks, parent, target, role = prepare_fake(bundle)
    p.fault = fault
    with pytest.raises(ValueError):
        runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    assert "solve" not in p.calls


@pytest.mark.parametrize("fault", ["clock_clipping", "counter_reset", "old_forcing", "output_history", "postsolve_remesh"])
def test_fixed_step_refuses_clipping_old_forcing_and_unqualified_outputs(bundle, fault):
    p, hooks, parent, target, role = prepare_fake(bundle)
    runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    p.fault = fault
    with pytest.raises(ValueError):
        runtime.fixed_steps(p, role, bundle.packet["controls"], bundle.packet["thresholds"], hooks,
                            datetime(2099, 1, 1, tzinfo=UTC))


def test_newton_and_deadline_budgets_stop_before_an_extra_update(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls = copy.deepcopy(bundle.packet["controls"])
    controls["max_total_newton_updates"] = 1
    runtime.start_jump(p, parent, target, role, controls, hooks)
    p.fault = "newton_budget"
    with pytest.raises(ValueError, match="Newton-update budget"):
        runtime.fixed_steps(p, role, controls, bundle.packet["thresholds"], hooks, datetime(2099, 1, 1, tzinfo=UTC))
    assert p.calls.count("solve") == 1
    with pytest.raises(ValueError, match="deadline elapsed"):
        runtime.fixed_steps(p, role, controls, bundle.packet["thresholds"], hooks, datetime(2000, 1, 1, tzinfo=UTC))
    assert p.calls.count("solve") == 1


def test_quality_remesh_is_between_native_rows_preserves_history_and_has_no_terminal_remesh(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls, thresholds = bundle.packet["controls"], bundle.packet["thresholds"]
    runtime.start_jump(p, parent, target, role, controls, hooks)
    p.remesh_requested = True
    guard = runtime.RemeshGuard(p, hooks, runtime.empty_work(), thresholds, controls)
    got = runtime.fixed_steps(p, role, controls, thresholds, hooks, datetime(2099, 1, 1, tzinfo=UTC), remesh_guard=guard)
    calls = [name for name in p.calls if name in {"solve", "output", "remesh"}]
    assert calls == ["solve", "output", "remesh", "solve", "output", "remesh", "solve", "output"]
    assert got["work"]["physical_remeshes"] == 2 and got["work"]["cumulative_transfer_relative"] == 0.0
    assert all(row["field_qualification"] == "interpolated_unqualified_before_next_step"
               for row in got["work"]["transfer_ledger"])
    assert got["terminal_witness"]["counter"] == 3 and p.remeshing_necessary()


@pytest.mark.parametrize("fault", ["flatten_history", "volume_transfer", "foreign_pending"])
def test_transfer_guards_refuse_flattening_cumulative_volume_and_foreign_domains(bundle, fault):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls, thresholds = bundle.packet["controls"], copy.deepcopy(bundle.packet["thresholds"])
    thresholds["cumulative_remesh_volume_relative"] = 7e-7
    if fault == "volume_transfer":
        thresholds["volume_relative"] = 2e-6  # Isolate the cumulative-transfer predicate in this software fixture.
    runtime.start_jump(p, parent, target, role, controls, hooks)
    p.remesh_requested = True
    p.fault = fault
    guard = runtime.RemeshGuard(p, hooks, runtime.empty_work(), thresholds, controls)
    if fault == "foreign_pending":
        p._domains_to_remesh = {Template(p)}
        with pytest.raises(ValueError, match="foreign/additional"):
            guard.owned()
    else:
        reason = "mapping norm budget" if fault == "flatten_history" else "absolute remesh volume transfer budget"
        with pytest.raises(ValueError, match=reason):
            runtime.fixed_steps(p, role, controls, thresholds, hooks, datetime(2099, 1, 1, tzinfo=UTC), remesh_guard=guard)


def test_load_rebuild_callback_is_administrative_and_unmatched_physical_callbacks_refuse(bundle):
    p, hooks, _parent, _target, _role = prepare_fake(bundle)
    work = runtime.empty_work()
    guard = runtime.RemeshGuard(p, hooks, work, bundle.packet["thresholds"], bundle.packet["controls"])
    with guard.installed():
        with guard.scope("administrative_load"):
            p.actions_after_remeshing()
        assert work == runtime.empty_work()
        with pytest.raises(ValueError, match="unmatched physical"):
            p.actions_after_remeshing()


def test_remesh_during_native_step_refuses_the_unqualified_recovery_adapter(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls, thresholds = bundle.packet["controls"], bundle.packet["thresholds"]
    runtime.start_jump(p, parent, target, role, controls, hooks)
    p.fault = "postsolve_remesh"
    guard = runtime.RemeshGuard(p, hooks, runtime.empty_work(), thresholds, controls)
    with pytest.raises(ValueError, match="inversion recovery transfer adapter is unqualified"):
        runtime.fixed_steps(p, role, controls, thresholds, hooks, datetime(2099, 1, 1, tzinfo=UTC), remesh_guard=guard)
    assert guard.work["physical_remeshes"] == 0


def test_dynamic_resume_restores_full_history_policy_and_continues_without_startup(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls, thresholds = bundle.packet["controls"], bundle.packet["thresholds"]
    runtime.start_jump(p, parent, target, role, controls, hooks)
    first_role = {**role, "max_steps": 1}
    first = runtime.fixed_steps(p, first_role, controls, thresholds, hooks, datetime(2099, 1, 1, tzinfo=UTC))
    saved = p.raw()
    cold = FakeProblem(target, saved)
    cold_hooks = fake_hooks(cold, bundle.binder)
    resumed = {"target_case": target, "witness": saved, "accepted_step_index": 1,
               "state": {"path": "/synthetic/state.dump"}, "mesh_exchange": {"mesh.msh": "/synthetic/mesh.msh"}}
    runtime.resume_jump(cold, resumed, role, controls, cold_hooks)
    assert cold.raw() == saved
    assert first["rows"][-1]["next_output_index"] == saved["policy"]["serialization"]["output_step"]
    assert cold.calls == ["load_state", "undo_steady_0", "undo_steady_1", "undo_steady_2"]
    got = runtime.fixed_steps(cold, role, controls, thresholds, cold_hooks, datetime(2099, 1, 1, tzinfo=UTC),
                              accepted_step_index=1, work=first["work"], retained_rows=first["rows"])
    assert cold.calls.count("solve") == 2 and got["work"]["solve_calls"] == 3
    assert got["terminal_witness"]["counter"] == 3 and [r["t_hat"] for r in got["rows"]] == [0, 50, 100, 150]
    assert [row["next_output_index"] for row in got["rows"]] == [4, 5, 6, 7]
    assert not any(name in cold.calls for name in ("set_time", "impulsive_fill", "reset_counter", "set_Ca_in"))
    bad = FakeProblem(target, saved)
    bad.fault = "load_history"
    with pytest.raises(ValueError, match="physical histories"):
        runtime.resume_jump(bad, resumed, role, controls, fake_hooks(bad))


def test_cold_resume_restores_owned_pending_quality_request_and_the_next_transition(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    controls, thresholds = bundle.packet["controls"], bundle.packet["thresholds"]
    runtime.start_jump(p, parent, target, role, controls, hooks)
    p.remesh_requested = True
    first = runtime.fixed_steps(p, {**role, "max_steps": 1}, controls, thresholds, hooks,
                                datetime(2099, 1, 1, tzinfo=UTC))
    saved = p.raw()
    assert saved["pending_quality_remesh"]["requested"] is True
    cold = FakeProblem(target, saved)
    cold.remesh_requested = True  # The same synthetic future quality rule.
    cold_hooks = fake_hooks(cold, bundle.binder)
    resumed = {"target_case": target, "witness": saved, "accepted_step_index": 1,
               "state": {"path": "/synthetic/state.dump"},
               "mesh_exchange": {"mesh.msh": "/synthetic/mesh.msh"}}
    runtime.resume_jump(cold, resumed, role, controls, cold_hooks)
    assert cold.raw() == saved and cold.remeshing_necessary()
    assert cold._domains_to_remesh == {cold._meshtemplate_list[0]._get_template()}
    receipts = []
    for problem, adapter in ((p, hooks), (cold, cold_hooks)):
        guard = runtime.RemeshGuard(problem, adapter, copy.deepcopy(first["work"]), thresholds, controls)
        receipts.append(runtime.fixed_steps(problem, {**role, "max_steps": 2}, controls, thresholds,
                        adapter, datetime(2099, 1, 1, tzinfo=UTC), accepted_step_index=1,
                        work=first["work"], retained_rows=first["rows"], remesh_guard=guard))
    assert [result["work"]["physical_remeshes"] for result in receipts] == [1, 1]
    assert receipts[0]["terminal_witness"] == receipts[1]["terminal_witness"]
    assert receipts[0]["rows"][-1]["metrics"] == receipts[1]["rows"][-1]["metrics"]


def test_pending_request_restoration_refuses_foreign_or_unrecorded_owners(bundle):
    p, hooks, parent, _target, _role = prepare_fake(bundle)
    p._domains_to_remesh = {Template(p)}
    with pytest.raises(ValueError, match="foreign/additional"):
        runtime.restore_pending_remesh(p, parent["witness"]["pending_quality_remesh"], hooks)
    p._domains_to_remesh = {p._meshtemplate_list[0]._get_template()}
    with pytest.raises(ValueError, match="unrecorded"):
        runtime.restore_pending_remesh(p, parent["witness"]["pending_quality_remesh"], hooks)
    policy = {**parent["witness"]["pending_quality_remesh"], "domains": ["foreign"]}
    with pytest.raises(ValueError, match="owned quality-remesh policy"):
        runtime.restore_pending_remesh(p, policy, hooks)


def test_fresh_jump_cannot_discard_a_parent_quality_request(bundle):
    p, hooks, parent, target, role = prepare_fake(bundle)
    parent["witness"]["pending_quality_remesh"]["requested"] = True
    with pytest.raises(ValueError, match="startup transfer is unqualified"):
        runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    assert not p.calls


@pytest.mark.parametrize("fault", ["missing_mesh", "missing_history", "missing_diagnostics", "old_control"])
def test_dynamic_resume_verifies_every_earlier_native_row_and_control_history(bundle, fault):
    resumed, study_ref, plan, role = dynamic_packet(bundle)
    ledger = json.loads(Path(resumed["step_ledger"]["path"]).read_text())
    first = ledger["rows"][0]
    if fault == "missing_mesh":
        (Path(first["state"]["path"]).parent / "mesh.msh").unlink()
    elif fault == "missing_history":
        Path(first["physical_witness"]["path"]).unlink()
    elif fault == "missing_diagnostics":
        first.pop("metrics")
    else:
        physical = json.loads(Path(first["physical_witness"]["path"]).read_text())
        physical["witness"]["parameters"]["Ca_in"] += 0.001
        first["physical_witness"] = bundle.binder.bind("changed-earlier-control.json", physical)
    resumed["step_ledger"] = bundle.binder.bind("changed-step-ledger.json", ledger)
    reference = bundle.binder.bind("changed-dynamic-resume.json", resumed)
    with pytest.raises(ValueError):
        cli.validate_dynamic_resume(reference, study_ref, bundle.packet, plan, role, study.Artifacts())


def test_native_retention_refuses_a_missing_portable_mesh_before_next_step(bundle):
    from dataclasses import replace

    p, hooks, parent, target, role = prepare_fake(bundle)
    runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    original = hooks.retain

    def incomplete(problem, row, witness):
        result = original(problem, row, witness)
        result.pop("mesh_inventory")
        return result

    with pytest.raises(ValueError, match="bound path and SHA-256"):
        runtime.fixed_steps(p, role, bundle.packet["controls"], bundle.packet["thresholds"],
                            replace(hooks, retain=incomplete), datetime(2099, 1, 1, tzinfo=UTC))
    assert "solve" not in p.calls


def test_physical_step_update_budget_survives_retries_and_cold_ledger_validation(bundle):
    controls = copy.deepcopy(bundle.packet["controls"])
    controls["max_newton_iterations_per_step"] = 1
    work = runtime.empty_work()
    budget = runtime.UpdateBudget(work, controls, datetime(2099, 1, 1, tzinfo=UTC))
    budget.before_solve()
    budget.before_update()
    budget.after_update()
    budget.after_solve()
    budget.before_solve()
    with pytest.raises(ValueError, match="Newton-update budget"):
        budget.before_update()
    assert work["newton_updates"] == 1
    with pytest.raises(ValueError, match="incomplete native Newton callback ledger"):
        budget.finish()
    resumed, _, _, _ = dynamic_packet(bundle)
    assert runtime.validate_work(resumed["work"], controls, bundle.packet["thresholds"], 1)["newton_step_updates"] == [1]
    drifted = {**resumed["work"], "newton_updates": 2, "newton_step_updates": [2]}
    with pytest.raises(ValueError, match="physical-step Newton-update budget"):
        runtime.validate_work(drifted, controls, bundle.packet["thresholds"], 1)


def callback_problem(*, failing_after_step=False):
    events = []

    def callback(name):
        def action():
            events.append(name)
            if name == "actions_after_newton_step" and failing_after_step:
                raise RuntimeError("synthetic after-update hook failure")
        return action

    return SimpleNamespace(events=events, **{name: callback(name) for name in runtime.NewtonCallbackBridge.CALLBACKS})


def test_native_callback_order_accepts_zero_update_initial_convergence(bundle):
    p = callback_problem()
    original = {name: getattr(p, name) for name in runtime.NewtonCallbackBridge.CALLBACKS}
    work, records = runtime.empty_work(), []
    budget = runtime.UpdateBudget(work, bundle.packet["controls"], datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda name, row: records.append((name, row)))
    with bridge.installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        p.events.append("initial_residual_already_converged")
        p.actions_after_newton_solve()
    budget.finish()
    assert work["newton_updates"] == 0 and work["newton_step_updates"] == [0]
    assert budget.callback_receipt()["zero_update_solves"] == 1
    assert not budget.callback_receipt()["unmatched_possible_update"]
    assert "actions_after_newton_step" not in p.events
    assert all(getattr(p, name) is action for name, action in original.items())
    assert records[-1][0] == "native_newton_call_completed"


def test_native_callback_bridge_stops_before_a_second_physical_update(bundle):
    p = callback_problem()
    controls = {**bundle.packet["controls"], "max_newton_iterations_per_step": 1}
    work, records = runtime.empty_work(), []
    budget = runtime.UpdateBudget(work, controls, datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda name, row: records.append((name, row)))
    with pytest.raises(ValueError, match="Newton-update budget"), bridge.installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        p.actions_after_newton_step()
        p.actions_before_newton_step()
    assert work["newton_updates"] == 1
    assert p.events.count("actions_before_newton_step") == 1
    assert records[-1][0] == "interrupted_native_newton_call"
    assert records[-1][1]["native_recovery_qualified"] is False
    with pytest.raises(ValueError, match="invalidated native callback accounting"):
        budget.finish()


def test_interrupted_initial_residual_is_not_a_completed_update_or_a_resumable_retry(bundle):
    p, records = callback_problem(), []
    work = runtime.empty_work()
    budget = runtime.UpdateBudget(work, bundle.packet["controls"], datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda name, row: records.append((name, row)))
    with pytest.raises(RuntimeError, match="synthetic residual failure"), bridge.installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        raise RuntimeError("synthetic residual failure")
    assert work["newton_updates"] == 0
    assert records[-1][1]["unmatched_possible_update"] is True
    assert records[-1][1]["counts_qualified_for_resume"] is False
    assert records[-1][1]["cumulative_work"]["newton_updates"] == 0
    assert records[-1][1]["cumulative_work"]["solve_calls"] == 0
    with pytest.raises(ValueError, match="invalidated native callback accounting"):
        budget.finish()
    with pytest.raises(ValueError, match="qualified recovery"):
        budget.before_solve()


def test_completed_update_is_counted_when_an_original_after_step_hook_fails(bundle):
    p, records = callback_problem(failing_after_step=True), []
    work = runtime.empty_work()
    budget = runtime.UpdateBudget(work, bundle.packet["controls"], datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda name, row: records.append((name, row)))
    with pytest.raises(RuntimeError, match="after-update hook failure"), bridge.installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        p.actions_after_newton_step()
    assert work["newton_updates"] == 1
    assert records[-1][1]["completed_updates"] == 1
    assert records[-1][1]["cumulative_work"]["newton_updates"] == 1
    assert records[-1][1]["cumulative_work"]["solve_calls"] == 0
    assert records[-1][1]["solve_completed"] is False
    assert records[-1][1]["native_recovery_qualified"] is False
    assert records[-1][1]["counts_qualified_for_resume"] is False
    with pytest.raises(ValueError, match="invalidated native callback accounting"):
        budget.finish()


@pytest.mark.parametrize("failing_hook", runtime.NewtonCallbackBridge.CALLBACKS)
def test_caught_original_callback_failure_stays_unqualified(bundle, failing_hook):
    p = callback_problem()
    original = getattr(p, failing_hook)

    def failure():
        original()
        raise RuntimeError("synthetic caught callback failure")

    setattr(p, failing_hook, failure)
    budget = runtime.UpdateBudget(runtime.empty_work(), bundle.packet["controls"],
                                  datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda *_: None)
    with pytest.raises(ValueError, match="invalidated native callback sequence"), bridge.installed():
        for name in bridge.CALLBACKS:
            try:
                getattr(p, name)()
            except RuntimeError:
                break
    receipt = budget.callback_receipt()
    assert receipt["observed_hooks"] and receipt["failure_reasons"]
    assert receipt["work_counts_qualified"] is False
    assert receipt["physical_completed_updates"] is None
    assert receipt["unmatched_possible_update"] is True
    assert receipt["counts_qualified_for_resume"] is False
    with pytest.raises(ValueError, match="invalidated native callback accounting"):
        budget.finish()


def test_caught_duplicate_after_solve_cannot_qualify_zero_update_work(bundle):
    p = callback_problem()
    budget = runtime.UpdateBudget(runtime.empty_work(), bundle.packet["controls"],
                                  datetime(2099, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="invalidated native callback sequence"), \
            runtime.NewtonCallbackBridge(p, budget, lambda *_: None).installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        p.actions_after_newton_solve()
        with pytest.raises(ValueError, match="unmatched after-Newton-solve"):
            p.actions_after_newton_solve()
    receipt = budget.callback_receipt()
    assert receipt["completed_updates"] == 0
    assert receipt["observed_hooks"].count("after_solve") == 2
    assert receipt["physical_completed_updates"] is None
    assert receipt["work_counts_qualified"] is False


def test_explicit_invalidated_matched_work_refuses_finish_before_debit(bundle):
    p = callback_problem()
    budget = runtime.UpdateBudget(runtime.empty_work(), bundle.packet["controls"],
                                  datetime(2099, 1, 1, tzinfo=UTC))
    with runtime.NewtonCallbackBridge(p, budget, lambda *_: None).installed():
        for name in runtime.NewtonCallbackBridge.CALLBACKS:
            getattr(p, name)()
    before = copy.deepcopy(budget.work)
    budget.invalidate("synthetic consumer rejects an otherwise matched sequence")
    with pytest.raises(ValueError, match="invalidated native callback accounting"):
        budget.finish()
    assert budget.work == before and budget.finished is False
    assert budget.callback_receipt()["completed_updates"] == 1
    assert budget.callback_receipt()["physical_completed_updates"] is None


def test_unmatched_after_step_retains_raw_zero_without_physical_zero_claim(bundle):
    p = callback_problem()
    budget = runtime.UpdateBudget(runtime.empty_work(), bundle.packet["controls"],
                                  datetime(2099, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="unmatched after-Newton-step"), \
            runtime.NewtonCallbackBridge(p, budget, lambda *_: None).installed():
        p.actions_after_newton_step()
    receipt = budget.callback_receipt()
    assert receipt["completed_updates"] == receipt["initial_residual_checks"] == 0
    assert receipt["observed_hooks"] == ["after_step"]
    assert receipt["callback_sequence_qualified"] is receipt["work_counts_qualified"] is False
    assert receipt["physical_completed_updates"] is None


def test_unfinished_native_solve_cannot_enter_an_unqualified_retry(bundle):
    p = callback_problem()
    budget = runtime.UpdateBudget(runtime.empty_work(), bundle.packet["controls"],
                                  datetime(2099, 1, 1, tzinfo=UTC))
    bridge = runtime.NewtonCallbackBridge(p, budget, lambda name, row: None)
    with pytest.raises(ValueError, match="interrupted native solve"), bridge.installed():
        p.actions_before_newton_solve()
        p.actions_before_newton_step()
        p.actions_after_newton_step()
        p.actions_before_newton_solve()
    assert budget.callback_receipt()["solve_attempts"] == 1


def test_native_capture_obligations_remain_an_unqualified_contract():
    contract = runtime.native_capture_contract()
    assert contract["pyoomph_pin"] == study.PYOOMPH_PIN
    assert contract["production_adapter_implemented"] is True
    assert contract["native_adapter_qualified"] is False
    assert "every actual ntstorage" in contract["history"]


def complete_stepper_clock(problem):
    """Model the observed complete native records, including dt-derived weights."""
    original = problem.raw

    def raw():
        witness = original()
        dt, previous = witness["dts"]
        nt = 7
        bdf1 = [[1.0] + [0.0] * (nt - 1), [1 / dt, -1 / dt] + [0.0] * (nt - 2), [0.0] * nt]
        bdf2 = [[1.0] + [0.0] * (nt - 1),
                [1 / dt + 1 / (dt + previous), -(dt + previous) / (dt * previous), dt / ((dt + previous) * previous)]
                + [0.0] * (nt - 3), [0.0] * nt]
        acceleration = [2 / (0.5 * dt * dt), -2 / (0.5 * dt * dt), 0.0, -2 / (dt * 0.5), -1.0, 0.0, 0.0]
        velocity = [0.5 * dt * acceleration[0], 0.5 * dt * acceleration[1], 0.0,
                    1 + 0.5 * dt * acceleration[3], dt * (1 - 0.5) + 0.5 * dt * acceleration[4], 0.0, 0.0]
        newmark = [[1.0] + [0.0] * (nt - 1), velocity, acceleration]
        for stepper in witness["steppers"]:
            stepper.update(time=witness["time"], dts=witness["dts"][:], counter=witness["counter"],
                           nprev_values=2, time_history=[witness["time"], witness["time"] - dt,
                                                       witness["time"] - dt - previous],
                           weights={"BDF1": copy.deepcopy(bdf1), "BDF2": copy.deepcopy(bdf2), "Newmark2": copy.deepcopy(newmark)})
        return witness

    problem.raw = raw


@pytest.mark.parametrize("parent_time", [8.0, 0.0])
def test_registered_startup_accepts_complete_clock_with_nonzero_origin_and_unequal_dt(bundle, parent_time):
    p, hooks, parent, target, role = prepare_fake(bundle)
    p.saved["time"] = p.witness["time"] = parent_time
    p.saved["dts"] = p.witness["dts"] = [0.25, 0.125]
    complete_stepper_clock(p)
    parent["witness"] = p.raw()
    current = [block["rows"][0][:] for block in parent["witness"]["data"]]
    got = runtime.start_jump(p, parent, target, role, bundle.packet["controls"], hooks)
    assert got["time"] == 0.0 and got["dts"] == [role["dt_solver"]] * 2
    assert [block["rows"][0] for block in got["data"]] == current
    assert got["steppers"][0]["time_history"] == [0.0, -role["dt_solver"], -2 * role["dt_solver"]]
    assert p.calls.count("load_state") == 1 and "solve" not in p.calls


@pytest.mark.parametrize("fault", ["registry", "counter", "history_clock", "weights"])
def test_clock_transition_refuses_changes_outside_the_declared_operation(bundle, fault):
    p, _hooks, _parent, _target, _role = prepare_fake(bundle)
    complete_stepper_clock(p)
    before = p.raw()
    after = copy.deepcopy(before)
    after["time"] = 0.0
    for stepper in after["steppers"]:
        stepper["time"] = 0.0
        stepper["time_history"] = [0.0, -0.25, -0.5]
    if fault == "registry":
        after["steppers"][0]["ntstorage"] = 5
    elif fault == "counter":
        after["steppers"][0]["counter"] += 1
    elif fault == "history_clock":
        after["steppers"][0]["time_history"][1] += 0.125
    else:
        after["steppers"][0]["weights"]["BDF1"][1][0] += 1.0
    with pytest.raises(ValueError):
        runtime.clock_transition(before, after)


def test_impulsive_dt_transition_refuses_finite_but_wrong_weights(bundle):
    p, _hooks, _parent, _target, role = prepare_fake(bundle)
    complete_stepper_clock(p)
    before = p.raw()
    p.assign_initial_values_impulsive(role["dt_solver"])
    after = p.raw()
    after["steppers"][0]["weights"]["BDF2"][1][1] += 1.0
    with pytest.raises(ValueError, match="inconsistent native weight"):
        runtime.clock_transition(before, after, dt=role["dt_solver"])


@pytest.mark.parametrize("fault", ["increment", "continue_counter", "history", "pins"])
def test_output_allows_only_the_proved_increment(bundle, fault):
    p, _hooks, _parent, _target, _role = prepare_fake(bundle)
    before = p.raw()
    p.output()
    after = p.raw()
    assert runtime.output_transition(before, after) == before["policy"]["serialization"]["output_step"] + 1
    if fault == "increment":
        after["policy"]["serialization"]["output_step"] += 1
    elif fault == "continue_counter":
        after["policy"]["serialization"]["continue_step"] += 1
    elif fault == "history":
        after["data"][0]["rows"][1][0] += 1
    else:
        after["data"][0]["pins"][0] = not after["data"][0]["pins"][0]
    with pytest.raises(ValueError, match="output changed"):
        runtime.output_transition(before, after)


def test_dynamic_resume_refuses_output_counter_drift(bundle):
    resumed, study_ref, plan, role = dynamic_packet(bundle)
    resumed["next_output_index"] += 1
    reference = bundle.binder.bind("wrong-output-resume.json", resumed)
    with pytest.raises(ValueError, match="next output index"):
        cli.validate_dynamic_resume(reference, study_ref, bundle.packet, plan, role, study.Artifacts())
