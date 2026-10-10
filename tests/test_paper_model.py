"""Software contracts for the isolated paper runner; no native model executes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from problems import paper_bubble
from runners import run_paper_case as runner

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "runners" / "run_paper_case.py"
PAPER_PROBLEMS = {"finite_gas_bubble", "bare_tube", "static_laplace"}
PAPER_CASE_DIRS = (
    ROOT / "validation" / "herradaEtAl_paper_model" / "cases",
    ROOT / "verification" / "paper_model" / "cases",
)


@pytest.fixture
def paper_case_path() -> Path:
    candidates = []
    for path in sorted(path for folder in PAPER_CASE_DIRS for path in folder.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("physics", {}).get("problem") in PAPER_PROBLEMS:
            candidates.append(path)
    assert candidates, "the paper model must provide a portable example case"
    paper_bubble.read_case(candidates[0])
    return candidates[0]


@pytest.fixture(autouse=True)
def forbid_native_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this software suite must never execute the native model")

    monkeypatch.setattr(runner, "run_model", forbidden)


@pytest.fixture
def clean_source_standin(monkeypatch) -> dict:
    """Stand in for a committed tree; this does not approve the live checkout."""
    head = "1" * 40
    revision = {
        "commit": head, "head": head, "clean": True, "status": [],
        "declared_commit": None, "sources": {},
    }
    monkeypatch.setattr(runner, "component_revision", lambda: copy.deepcopy(revision))
    return revision


def refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(arguments)
    assert error.value.code == 2


def events(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / runner.PROGRESS).read_text().splitlines()]


def test_check_needs_no_environment_or_clean_source(paper_case_path, tmp_path, monkeypatch):
    def forbidden_provenance():
        raise AssertionError("--check must stop before execution provenance checks")

    monkeypatch.setattr(runner, "component_revision", forbidden_provenance)
    monkeypatch.setattr(runner, "pyoomph_revision", forbidden_provenance)
    assert runner.main([str(paper_case_path), "--check"]) == 0
    out = tmp_path / "new-parent" / "output"
    assert runner.main([str(paper_case_path), "--out", str(out), "--check"]) == 0
    assert not out.parent.exists()


@pytest.mark.parametrize("working_directory", ["foreign", "runner"])
def test_check_imports_from_foreign_cwd_without_pyoomph(
    paper_case_path, tmp_path, working_directory,
):
    code = """
import runpy
import sys

class RefuseNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph."):
            raise AssertionError("--check imported pyoomph")

sys.meta_path.insert(0, RefuseNative())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    cwd = tmp_path if working_directory == "foreign" else RUNNER.parent
    result = subprocess.run(
        [sys.executable, "-c", code, str(RUNNER), str(paper_case_path), "--check"],
        cwd=cwd, env=environment, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "checked"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [[], ["--not-an-option"], ["--ou", "unused"]])
def test_missing_output_and_invalid_options_create_nothing(paper_case_path, tmp_path, arguments):
    refused([str(paper_case_path), *arguments])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("contents", ["{invalid JSON", '{"schema":"wrong"}'])
def test_invalid_case_creates_no_output(tmp_path, contents):
    case_path = tmp_path / "invalid.json"
    case_path.write_text(contents, encoding="utf-8")
    out = tmp_path / "missing-parent" / "output"
    refused([str(case_path), "--out", str(out)])
    assert not out.parent.exists()


@pytest.mark.parametrize("check", [False, True])
def test_occupied_output_is_unchanged(paper_case_path, tmp_path, check):
    out = tmp_path / "occupied"
    out.mkdir()
    sentinel = out / "sentinel.bin"
    sentinel.write_bytes(b"preserve this existing run\x00\xff")
    before = sentinel.read_bytes()
    refused([str(paper_case_path), "--out", str(out), *(["--check"] if check else [])])
    assert list(out.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == before


def test_component_output_and_symlink_alias_are_refused(paper_case_path, tmp_path):
    alias = tmp_path / "component-alias"
    alias.symlink_to(ROOT, target_is_directory=True)
    for out in (ROOT / "paper-contract-output", alias / "paper-contract-output"):
        refused([str(paper_case_path), "--out", str(out), "--check"])
        assert not out.exists()


def test_output_file_or_file_parent_is_refused(paper_case_path, tmp_path):
    sentinel = tmp_path / "file"
    sentinel.write_bytes(b"keep")
    for out in (sentinel, sentinel / "output"):
        refused([str(paper_case_path), "--out", str(out), "--check"])
    assert sentinel.read_bytes() == b"keep"


def test_dirty_and_uncommitted_methods_block_before_output(paper_case_path, tmp_path, monkeypatch):
    head = "2" * 40
    monkeypatch.setattr(runner, "component_revision", lambda: {
        "commit": f"{head}-dirty", "head": head, "clean": False,
        "status": ["?? problems/paper_bubble.py"],
        "sources": {"problems/paper_bubble.py": {"committed_matches": False}},
    })
    out = tmp_path / "not-created"
    refused([str(paper_case_path), "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("fault", ["wrong_commit", "missing_distribution", "missing_native_digest"])
def test_environment_faults_block_before_output(
    paper_case_path, tmp_path, monkeypatch, clean_source_standin, fault,
):
    revision = {
        "available": True, "commit": paper_bubble.PYOOMPH_COMMIT,
        "url": "https://github.com/comphy-lab/pyoomph",
        "direct_url": {"vcs_info": {"vcs": "git"}},
        "native_core": {"path": "native-core.so", "sha256": "3" * 64},
    }
    if fault == "wrong_commit":
        revision["commit"] = "0" * 40
    elif fault == "missing_distribution":
        revision["available"] = False
    else:
        revision["native_core"].pop("sha256")
    monkeypatch.setattr(runner, "pyoomph_revision", lambda: revision)
    out = tmp_path / "not-created"
    refused([str(paper_case_path), "--out", str(out)])
    assert not out.exists()


def test_manifest_precedes_progress_and_binds_actual_inputs(
    paper_case_path, tmp_path, monkeypatch, clean_source_standin,
):
    out = tmp_path / "fresh"
    real_progress = runner.append_progress
    order = []

    def progress(path, payload, *, first=False):
        assert (path / runner.MANIFEST).is_file()
        if first:
            assert sorted(item.name for item in path.iterdir()) == [runner.MANIFEST]
        order.append(payload["event"])
        return real_progress(path, payload, first=first)

    def model_standin(case, path):
        assert case == paper_bubble.read_case(paper_case_path)
        assert path == out
        assert [row["event"] for row in events(path)] == ["start"]
        order.append("model_standin")
        return {"status": "completed", "completion_reached": True, "scientific_status": "standin"}

    monkeypatch.setattr(runner, "append_progress", progress)
    monkeypatch.setattr(runner, "run_model", model_standin)
    arguments = [str(paper_case_path), "--out", str(out)]
    assert runner.main(arguments) == 0
    assert order == ["start", "model_standin", "final"]
    manifest_path = out / runner.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "pyoomph-runner-manifest-v1"
    assert manifest["case_file"] == manifest["case_path"]
    assert manifest["case_path"] == str(paper_case_path.resolve())
    assert manifest["case_sha256"] == hashlib.sha256(paper_case_path.read_bytes()).hexdigest()
    assert manifest["case_id"] == paper_bubble.read_case(paper_case_path)["case_id"]
    assert manifest["case_fields_used"] == list(paper_bubble.CASE_FIELDS_USED)
    assert manifest["units"] == "nondimensional (sigma=mu_liquid=R0=1)"
    assert manifest["nondimensionalization"] == paper_bubble.read_case(paper_case_path)["nondimensionalization"]
    assert manifest["runner"] == {"path": str(RUNNER),
                                  "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest()}
    assert manifest["runner_path"] == manifest["runner"]["path"]
    assert manifest["runner_sha256"] == manifest["runner"]["sha256"]
    assert manifest["runner_sha256"] == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert manifest["component_commit"] == clean_source_standin["commit"]
    assert manifest["component"]["clean"] is True
    assert manifest["pyoomph"]["commit"] == paper_bubble.PYOOMPH_COMMIT
    assert manifest["pyoomph"]["native_core"]["sha256"]
    assert manifest["pyoomph"]["interpreter"]["executable"] == sys.executable
    assert manifest["argv"] == [sys.executable, str(RUNNER), *arguments]
    before = manifest_path.read_bytes()
    refused(arguments)
    assert manifest_path.read_bytes() == before
    assert order == ["start", "model_standin", "final"]


@pytest.mark.parametrize("summary", [
    {"status": "completed", "completion_reached": False},
    {"status": "incomplete", "completion_reached": True},
    {"status": "completed", "completion_reached": 1},
])
def test_unreached_completion_preserves_summary_and_returns_failure(
    paper_case_path, tmp_path, monkeypatch, clean_source_standin, summary,
):
    monkeypatch.setattr(runner, "run_model", lambda case, path: summary)
    out = tmp_path / "incomplete"
    out.mkdir()  # An existing empty attempt directory is allowed.
    assert runner.main([str(paper_case_path), "--out", str(out)]) == 1
    assert json.loads((out / "summary.json").read_text()) == summary
    assert [row["event"] for row in events(out)] == ["start", "final"]
    assert events(out)[-1]["status"] == "incomplete"
    assert events(out)[-1]["exit_status"] == 1


def test_model_exception_is_recorded_after_manifest(
    paper_case_path, tmp_path, monkeypatch, clean_source_standin,
):
    def model_standin(case, out):
        raise RuntimeError("deliberate stand-in failure")

    monkeypatch.setattr(runner, "run_model", model_standin)
    out = tmp_path / "failed"
    assert runner.main([str(paper_case_path), "--out", str(out)]) == 1
    assert (out / runner.MANIFEST).is_file()
    assert not (out / "summary.json").exists()
    assert [row["event"] for row in events(out)] == ["start", "failure"]
    assert events(out)[-1]["error_type"] == "RuntimeError"
    assert events(out)[-1]["error"] == "deliberate stand-in failure"


def test_manifest_creation_is_exclusive_and_private(tmp_path):
    runner.create_run_manifest(tmp_path, {"schema": "software-standin"})
    path = tmp_path / runner.MANIFEST
    before = path.read_bytes()
    assert stat.S_ISREG(path.stat().st_mode)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        runner.create_run_manifest(tmp_path, {"schema": "replacement"})
    assert path.read_bytes() == before


def test_manifest_never_follows_an_existing_symlink(tmp_path):
    target = tmp_path / "preserve.json"
    target.write_bytes(b"preserve")
    out = tmp_path / "output"
    out.mkdir()
    (out / runner.MANIFEST).symlink_to(target)
    with pytest.raises(FileExistsError):
        runner.create_run_manifest(out, {"schema": "replacement"})
    assert target.read_bytes() == b"preserve"


def test_component_revision_truthfully_records_the_live_source():
    revision = runner.component_revision()
    assert revision["head"] and len(revision["head"]) == 40
    for relative in runner.METHOD_FILES:
        assert revision["sources"][relative]["sha256"] == runner.sha256_file(ROOT / relative)
    if revision["status"] or any(
        not source["committed_matches"] for source in revision["sources"].values()
    ):
        assert revision["clean"] is False
        assert revision["commit"] == f"{revision['head']}-dirty"
        with pytest.raises(ValueError, match="clean, committed"):
            runner.require_clean_source(revision)


def test_installed_pin_native_digest_and_interpreter_need_no_solver_import():
    imported_before = {name for name in sys.modules if name == "pyoomph" or name.startswith("pyoomph.")}
    revision = runner.pyoomph_revision()
    runner.require_pinned_environment(revision)
    assert revision["commit"] == paper_bubble.PYOOMPH_COMMIT
    assert revision["direct_url"]["vcs_info"]["commit_id"] == paper_bubble.PYOOMPH_COMMIT
    assert revision["native_core"]["sha256"] == runner.sha256_file(Path(revision["native_core"]["path"]))
    assert revision["interpreter"]["executable"] == sys.executable
    imported_after = {name for name in sys.modules if name == "pyoomph" or name.startswith("pyoomph.")}
    assert imported_after == imported_before


def paper_cases():
    return sorted((ROOT / "verification/paper_model/cases").glob("*.json")) + sorted(
        (ROOT / "validation/herradaEtAl_paper_model/cases").glob("*.json")
    )


@pytest.mark.parametrize("path", paper_cases(), ids=lambda path: path.stem)
def test_every_new_case_has_valid_units_ports_and_geometry(path):
    import math

    case = paper_bubble.read_case(path)
    geom = paper_bubble.case_geometry(case)
    assert case["physics"]["reynolds"] == 0
    assert case["physics"]["gas_density_ratio"] == 0.000949
    assert case["physics"]["gas_viscosity_ratio"] == 0.000018
    assert geom.l_domain / geom.r_tube == pytest.approx(case["physics"]["domain_over_r"])
    if case["physics"]["problem"] != "bare_tube":
        actual = 4 * math.pi * geom.r_body**3 / 3 + math.pi * geom.r_body**2 * (
            geom.z_front - geom.z_rear - 2 * geom.r_body
        )
        assert actual == pytest.approx(4 * math.pi / 3)
        assert geom.z_front / geom.r_tube == pytest.approx(case["physics"]["upper_tip_over_r"])
        assert (0 - geom.z_front) / geom.r_tube == pytest.approx(-2 * case["physics"]["domain_over_r"] / 3)
        assert (geom.l_domain - geom.z_front) / geom.r_tube == pytest.approx(case["physics"]["domain_over_r"] / 3)


@pytest.mark.parametrize("section,name,value,match", [
    ("geometry", "frame", "laboratory", "geometry"),
    ("geometry", "coordinates", ["z", "r"], "geometry"),
    ("nondimensionalization", "liquid_viscosity", 2.0, "nondimensionalization"),
    ("nondimensionalization", "surface_tension", True, "surface_tension"),
    ("physics", "reynolds", 1.0, "reynolds"),
    ("physics", "problem", [], "physics.problem"),
    ("physics", "gas_density_ratio", 0.0, "gas_density_ratio"),
    ("physics", "gas_viscosity_ratio", 0.0, "gas_viscosity_ratio"),
    ("physics", "gas_viscosity_ratio", 0.000949, "gas_viscosity_ratio"),
    ("physics", "upper_tip_over_r", 9.0, "upper_tip_over_r"),
    ("physics", "vhat", 3.0, "vhat"),
    ("pressure", "field", "reduced", "pressure"),
    ("pressure", "body_force", "liquid_only", "pressure"),
    ("boundaries", "lower", "traction_free", "boundaries"),
    ("boundaries", "lower_tangential", "zero_traction", "boundaries"),
    ("closure", "frame_speed", "centroid", "closure"),
    ("closure", "gas_volume", "uniform_void_pressure", "closure"),
    ("mesh", "template", "LiquidAnnulusMesh", "mesh.template"),
    ("mesh", "film_elements", 0.5, "film_elements"),
    ("mesh", "bulk_resolution", float("nan"), "bulk_resolution"),
    ("mesh", "remeshing", 1, "remeshing"),
    ("solve", "max_bond_steps", True, "max_bond_steps"),
    ("solve", "max_bond_steps", 1, "cannot reach"),
    ("physics", "laplace", 1.0, "unused fields"),
    ("mesh", "gas_resolution", 0.2, "unused fields"),
    ("completion", "value", "process_exited", "completion"),
])
def test_wrong_physics_or_port_contract_is_refused(section, name, value, match):
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    case[section][name] = value
    with pytest.raises(ValueError, match=match):
        paper_bubble.validate_case(case)


def test_validator_reports_multiple_errors_without_mutating_input():
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    case["physics"]["reynolds"] = 3
    case["pressure"]["field"] = "reduced"
    case["mesh"]["film_elements"] = 0
    original = copy.deepcopy(case)
    with pytest.raises(ValueError) as error:
        paper_bubble.validate_case(case)
    assert all(text in str(error.value) for text in ("reynolds", "pressure", "film_elements"))
    assert case == original


def test_density_ratio_enters_stokes_buoyancy_once_and_viscosity_stays_finite():
    radius = 3.3**(-1 / 3)
    liquid = paper_bubble.phase_force_density(1.56, radius)
    gas = paper_bubble.phase_force_density(1.56, radius, 0.000949)
    assert liquid == pytest.approx((0, -3.4578054802295966))
    assert gas == pytest.approx((0, -0.0032814574007378867))
    assert gas[1] / liquid[1] == pytest.approx(0.000949)
    assert 1.56 * (1 - 0.000949) == pytest.approx(1.55851956)
    with pytest.raises(ValueError, match="positive"):
        paper_bubble.phase_force_density(1.56, radius, 0)


def test_independent_bare_tube_reference_obeys_flow_pressure_and_frame_mapping():
    radius, ca, bond, speed, height = 0.8, -0.003, 1.56, 0.02, 12.0
    wall = paper_bubble.analytic_tube_fields(radius, height, radius=radius,
        ca_in=ca, bond=bond, frame_speed=speed)
    lower = paper_bubble.analytic_tube_fields(0, 0, radius=radius,
        ca_in=ca, bond=bond, frame_speed=speed)
    assert wall["velocity_z"] == pytest.approx(-speed)
    assert lower["pressure_static"] == 0
    assert lower["velocity_z"] == pytest.approx(2 * ca - speed)
    # Exact integration of a parabola gives the mean laboratory velocity Ca.
    mid = paper_bubble.analytic_tube_fields(radius / 2**0.5, 0, radius=radius,
        ca_in=ca, bond=bond, frame_speed=speed)
    assert mid["velocity_z"] + speed == pytest.approx(ca)
    # The hydraulic gradient and gravitational potential both enter pressure.
    pressure_gradient = wall["pressure_static"] / height
    assert -pressure_gradient - 8 * ca / radius**2 - bond / radius**2 == pytest.approx(0)


def recording_mesh(case, *, remeshing=False, live_polyline=None):
    """Exercise the geometry declarations without Gmsh or native assembly."""
    from types import SimpleNamespace

    from problems.paper_mesh import PaperTwoPhaseMesh

    class RecordingMesh:
        _interface = PaperTwoPhaseMesh._interface
        _graded_size = PaperTwoPhaseMesh._graded_size

        def __init__(self):
            self.gmsh_options = {}
            self.curves = []
            self.surfaces = []
            self.problem = SimpleNamespace(geometry=paper_bubble.case_geometry(case),
                bulk_resolution=case["mesh"]["bulk_resolution"],
                film_elements=case["mesh"]["film_elements"])

        def get_problem(self):
            return self.problem

        def is_remeshing(self):
            return remeshing

        def get_boundary_coordinates(self, name, **kwargs):
            assert name == "liquid/interface"
            return [self.problem.geometry.interface_polyline() if live_polyline is None else live_polyline]

        def point(self, r, z):
            return float(r), float(z)

        def line(self, first, second, *, name):
            assert first != second, "a spherical seed must not add a zero-length cylinder"
            self.curves.append((name, "line", first, second))

        def circle_arc(self, first, second, *, center, name):
            self.curves.append((name, "arc", first, second))

        def spline(self, points, *, name):
            self.curves.append((name, "spline", points[0], points[-1]))

        def plane_surface(self, *boundaries, name):
            self.surfaces.append((name, boundaries))

    mesh = RecordingMesh()
    PaperTwoPhaseMesh.define_geometry(mesh)
    return mesh


@pytest.mark.parametrize("case_path", [
    "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json",
    "verification/paper_model/cases/static-laplace.json",
])
def test_two_domains_reuse_one_interface_and_boundary_names_after_remesh(case_path):
    from problems.paper_mesh import GAS_BOUNDARIES, LIQUID_BOUNDARIES

    case = paper_bubble.read_case(ROOT / case_path)
    initial = recording_mesh(case)
    rebuilt = recording_mesh(case, remeshing=True)
    expected = [("liquid", LIQUID_BOUNDARIES), ("gas", GAS_BOUNDARIES)]
    assert initial.surfaces == rebuilt.surfaces == expected
    assert {name for name, *_ in initial.curves} == {name for name, *_ in rebuilt.curves}
    assert [name for name, *_ in initial.curves].count("gas_axis") == 1
    assert [name for name, *_ in rebuilt.curves].count("interface") == 1
    assert next(item[1] for item in rebuilt.curves if item[0] == "interface") == "spline"


def test_spherical_seed_has_no_zero_length_body_edge():
    case = paper_bubble.read_case(ROOT / "verification/paper_model/cases/static-laplace.json")
    mesh = recording_mesh(case)
    interface = [entry for entry in mesh.curves if entry[0] == "interface"]
    assert len(interface) == 2 and all(entry[1] == "arc" for entry in interface)


def test_remesh_floor_follows_the_live_thin_film_instead_of_the_seed_gap():
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    geom = paper_bubble.case_geometry(case)
    live_gap = 0.01
    scale = (geom.r_tube - live_gap) / geom.r_body
    live = [(float(r) * scale, float(z)) for r, z in geom.interface_polyline()]
    initial = recording_mesh(case)
    remeshed = recording_mesh(case, remeshing=True, live_polyline=live)
    assert remeshed._interface_size == pytest.approx(live_gap / case["mesh"]["film_elements"])
    assert remeshed._interface_size / initial._interface_size == pytest.approx(live_gap / geom.seed_gap)
    assert remeshed.gmsh_options["Mesh.MeshSizeMin"] == pytest.approx(remeshed._interface_size / 2)


@pytest.mark.parametrize("radius", [0.6716793862139855, 0.8, float("nan")])
def test_remeshing_refuses_contact_negative_or_nonfinite_wall_gap(radius):
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    geom = paper_bubble.case_geometry(case)
    live = [(0, geom.z_rear), (radius, geom.z_centroid), (0, geom.z_front)]
    with pytest.raises(ValueError, match="finite|positive"):
        recording_mesh(case, remeshing=True, live_polyline=live)


def test_nonzero_subresolution_cylinder_seed_is_refused_before_meshing():
    case = paper_bubble.read_case(ROOT / "verification/paper_model/cases/static-laplace.json")
    case["physics"]["seed_radius_over_tube"] = (1 - 1e-13) / case["physics"]["r_tube"]
    assert 0 < paper_bubble.case_geometry(case).cylinder_length < 1e-12
    with pytest.raises(ValueError, match="seed cylinder"):
        paper_bubble.validate_case(case)


@pytest.mark.parametrize("value", [True, float("nan"), -1, 2, "zero"])
def test_invalid_bond_override_is_refused_before_native_api_loading(value, monkeypatch):
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    monkeypatch.setattr(paper_bubble, "_native_problem_types",
                        lambda: pytest.fail("invalid override must not load the native APIs"))
    with pytest.raises(ValueError, match="initial_bond"):
        paper_bubble.build_problem(case, initial_bond=value)


def test_pinned_native_api_definitions_import_without_initialising_problem():
    from pyoomph.equations.navier_stokes import StokesEquations

    problem_type, pressure_type = paper_bubble._native_problem_types()
    assert problem_type.__name__ == "PaperBubbleProblem"
    assert pressure_type.required_parent_type is StokesEquations


@pytest.fixture
def official_project_helper() -> Path:
    configured = os.environ.get("PYOOMPH_PROJECT_HELPER")
    path = (Path(configured).expanduser() if configured else
            Path.home() / ".agents/skills/pyoomph-projects-comphy/scripts/pyoomph_project.py")
    assert path.is_file(), "install pyoomph-projects-comphy or set PYOOMPH_PROJECT_HELPER"
    return path.resolve()


def call_official_helper(helper: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Execute the installed validator itself, with no helper stand-in."""
    return subprocess.run(
        [sys.executable, str(helper), *arguments],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, check=False, timeout=30,
    )


@pytest.mark.parametrize("path", paper_cases(), ids=lambda path: path.stem)
def test_official_helper_validates_every_paper_case(official_project_helper, path):
    result = call_official_helper(official_project_helper, "validate-case", "--case", str(path))
    assert result.returncode == 0, result.stderr
    validated = json.loads(result.stdout)
    assert validated["schema"] == "pyoomph-case-v1"
    assert validated["case_id"] == paper_bubble.read_case(path)["case_id"]
    assert validated["path"] == str(path.resolve())
    assert validated["sha256"] == runner.sha256_file(path)


@pytest.fixture
def software_closeout_bundle(
    paper_case_path, tmp_path, monkeypatch, clean_source_standin, official_project_helper,
):
    """Bind actual files/environment metadata under synthetic source/solve gates.

    This is a validator fixture, not a launch receipt or numerical result.
    """
    monkeypatch.setattr(runner, "run_model", lambda case, path: {
        "status": "completed", "completion_reached": True,
        "scientific_status": "software_fixture_only",
    })
    runtime = tmp_path / "runtime"
    assert runner.main([str(paper_case_path), "--out", str(runtime)]) == 0
    manifest = json.loads((runtime / runner.MANIFEST).read_text())
    environment = manifest["pyoomph"]
    receipt = {
        "schema": "pyoomph-environment-v1", "compatible": True,
        "purpose": "software validator fixture; no native compatibility claim",
        "interpreter": environment["interpreter"],
        "pyoomph": {
            "resolved_commit": environment["commit"],
            "locked_commit": environment["commit"],
            "direct_url": environment["direct_url"],
            "core_sha256": environment["native_core"]["sha256"],
            "core_path": environment["native_core"]["path"],
        },
    }
    receipt_path = tmp_path / "software-environment.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    launch = {
        "schema": "pyoomph-run-manifest-v1", "created_at": manifest["created_at"],
        "case_id": manifest["case_id"],
        "case": {"path": str(paper_case_path), "sha256": runner.sha256_file(paper_case_path)},
        "runner": {"path": str(RUNNER), "sha256": runner.sha256_file(RUNNER)},
        "fem_commit": clean_source_standin["commit"],
        "pyoomph_commit": environment["commit"],
        "core_sha256": environment["native_core"]["sha256"],
        "environment_receipt": {"path": str(receipt_path),
                                "sha256": runner.sha256_file(receipt_path)},
    }
    launch_path = tmp_path / "software-launch-manifest.json"
    launch_path.write_text(json.dumps(launch) + "\n", encoding="utf-8")
    validation = call_official_helper(
        official_project_helper, "validate-manifest", "--manifest", str(launch_path),
    )
    assert validation.returncode == 0, validation.stderr
    return runtime, launch_path, receipt_path, manifest


def test_official_helper_accepts_runner_closeout_against_launch(
    official_project_helper, software_closeout_bundle,
):
    runtime, launch_path, receipt_path, manifest = software_closeout_bundle
    result = call_official_helper(
        official_project_helper, "validate-run-output", "--run", str(runtime),
        "--launch-manifest", str(launch_path),
    )
    assert result.returncode == 0, result.stderr
    closeout = json.loads(result.stdout)
    assert closeout["schema"] == "pyoomph-run-output-validation-v1"
    assert closeout["case_id"] == manifest["case_id"]
    assert closeout["component_commit"] == manifest["component_commit"]
    assert closeout["pyoomph_commit"] == paper_bubble.PYOOMPH_COMMIT
    assert closeout["launch_manifest"] == {
        "path": str(launch_path), "sha256": runner.sha256_file(launch_path),
    }
    assert closeout["claim"] == "provenance agreement only; no numerical verification or validation"
    assert json.loads(receipt_path.read_text())["pyoomph"]["core_sha256"] == manifest["pyoomph"]["native_core"]["sha256"]


def test_official_closeout_refuses_legacy_aliases_without_canonical_runner(
    official_project_helper, software_closeout_bundle,
):
    runtime, launch_path, _, manifest = software_closeout_bundle
    manifest.pop("runner")  # Deliberately spoil only this software fixture.
    (runtime / runner.MANIFEST).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    result = call_official_helper(
        official_project_helper, "validate-run-output", "--run", str(runtime),
        "--launch-manifest", str(launch_path),
    )
    assert result.returncode == 2
    assert "runner hash differs or is not recorded" in result.stderr


@pytest.mark.parametrize("cadence", [0, -1, True, float("nan"), float("inf")])
def test_output_cadence_requires_a_finite_positive_number(paper_case_path, cadence):
    case = paper_bubble.read_case(paper_case_path)
    case["output"]["cadence"] = cadence
    with pytest.raises(ValueError, match="cadence"):
        paper_bubble.validate_case(case)


def test_run_model_passes_case_cadence_to_transient_standin(tmp_path, monkeypatch):
    from types import ModuleType

    case = paper_bubble.read_case(ROOT / "verification/paper_model/cases/bare-tube.json")
    case["solve"]["relax_time"] = 8.0
    case["output"]["cadence"] = 0.4
    transient_calls = []
    solver_calls = []

    class ProblemStandin:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_output_directory(self, path):
            assert Path(path) == tmp_path / "solver"

        def set_linear_solver(self, name):
            assert name == "superlu"

        def set_c_compiler(self, name):
            assert name == "system"

        def initialise(self):
            solver_calls.append("initialise_standin")

        def run(self, duration, **kwargs):
            transient_calls.append((duration, kwargs))

        def solve(self, **kwargs):
            solver_calls.append(("solve_standin", kwargs))

        def observable(self, name, domain="liquid"):
            return 1.0 if name in ("volume", "area", "velocity_reference_squared",
                                  "pressure_reference_squared") else 0.0

        def frame_speed(self):
            return 0.0

        def save_state(self, path, *, quiet):
            assert quiet is True
            state = Path(path)
            state.parent.mkdir()
            state.write_bytes(b"software state stand-in")

    def build_standin(model_case, *, initial_bond):
        assert model_case == case
        assert initial_bond == case["solve"]["initial_bond"]
        return ProblemStandin()

    def copy_state_standin(source, destination, *, copy_mesh_files):
        assert copy_mesh_files is True
        Path(destination).write_bytes(Path(source).read_bytes())

    # The execution function's import resolves to this pure module, not pyoomph.
    package = ModuleType("pyoomph")
    package.__path__ = []
    output_package = ModuleType("pyoomph.output")
    output_package.__path__ = []
    states = ModuleType("pyoomph.output.states")
    states.copy_state_file = copy_state_standin
    for name, module in (("pyoomph", package), ("pyoomph.output", output_package),
                         ("pyoomph.output.states", states)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(paper_bubble, "build_problem", build_standin)
    monkeypatch.setattr(paper_bubble, "_stationary_residual", lambda problem: 0.0)
    runner.create_run_manifest(tmp_path, {"purpose": "software cadence fixture only"})
    summary = paper_bubble.run_model(case, tmp_path)
    assert transient_calls == [(8.0, {
        "startstep": case["solve"]["start_step"], "outstep": 0.4, "maxstep": 1.0,
        "temporal_error": case["tolerances"]["temporal_error"],
    })]
    assert solver_calls == ["initialise_standin", ("solve_standin", {
        "max_newton_iterations": case["solve"]["max_newton_iterations"],
    })]
    assert summary["status"] == "completed"
    assert summary["completion_reached"] is True
    assert (tmp_path / summary["checkpoint"]).read_bytes() == b"software state stand-in"


@pytest.mark.parametrize("integral,volume,expected", [(36.0, 4.0, 3.0), (0.0, 7.0, 0.0),
                                                     (4e-18, 4.0, 1e-9)])
def test_bulk_velocity_rms_uses_volume_weighted_integrals(integral, volume, expected):
    assert paper_bubble.bulk_velocity_rms(integral, volume) == pytest.approx(expected)


@pytest.mark.parametrize("integral,volume", [(-1e-18, 2.0), (1.0, 0.0), (float("nan"), 1.0),
                                           (1.0, float("inf"))])
def test_invalid_velocity_integrals_cannot_pass_as_zero_flow(integral, volume):
    with pytest.raises(ValueError, match="velocity RMS"):
        paper_bubble.bulk_velocity_rms(integral, volume)


@pytest.mark.parametrize("tolerance", [0, -1, True, float("nan"), float("inf")])
def test_static_velocity_tolerance_is_finite_and_positive(paper_case_path, tolerance):
    case = paper_bubble.read_case(paper_case_path)
    case["tolerances"]["static_velocity_absolute"] = tolerance
    with pytest.raises(ValueError, match="static_velocity_absolute"):
        paper_bubble.validate_case(case)


@pytest.fixture
def phase_velocity_summary_standin(tmp_path, monkeypatch):
    """Run actual summary predicates with pure phase-integral and solver stand-ins."""
    from types import ModuleType, SimpleNamespace

    import numpy as np

    def execute(case_path, liquid_rms, gas_rms, frame_speed):
        case = paper_bubble.read_case(ROOT / case_path)
        geom = paper_bubble.case_geometry(case)
        gas_volume = getattr(geom, "volume", 4.0)
        liquid_volume = 9.0

        class ProblemStandin:
            geometry = geom

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def set_output_directory(self, path):
                assert Path(path) == tmp_path / "solver"

            def set_linear_solver(self, name):
                assert name == "superlu"

            def set_c_compiler(self, name):
                assert name == "system"

            def initialise(self):
                pass

            def run(self, *args, **kwargs):
                pass

            def solve(self, **kwargs):
                pass

            def get_global_parameter(self, name):
                return SimpleNamespace(value=case["solve"]["initial_bond"])

            def get_ode(self, domain):
                assert domain == "gas_volume"
                return SimpleNamespace(get_value=lambda *args, **kwargs: 0.0)

            def frame_speed(self):
                return frame_speed

            def observable(self, name, domain="liquid"):
                if name == "volume":
                    return gas_volume if domain == "gas" else liquid_volume
                if name == "velocity_squared":
                    return gas_rms**2 * gas_volume if domain == "gas" else liquid_rms**2 * liquid_volume
                if name == "surface_volume":
                    return gas_volume
                if name == "pressure":
                    return 2 * gas_volume if domain == "gas" else 0.0
                return 1.0 if name in ("area", "velocity_reference_squared",
                                      "pressure_reference_squared") else 0.0

            def interface_polyline(self):
                return np.asarray([geom.z_front - 2, geom.z_front - 1, geom.z_front]), np.asarray([0, 1, 0])

            def save_state(self, path, **kwargs):
                path = Path(path)
                path.parent.mkdir()
                path.write_bytes(b"phase velocity software stand-in")

        def copy_state(source, destination, **kwargs):
            Path(destination).write_bytes(Path(source).read_bytes())

        package = ModuleType("pyoomph")
        package.__path__ = []
        output_package = ModuleType("pyoomph.output")
        output_package.__path__ = []
        states = ModuleType("pyoomph.output.states")
        states.copy_state_file = copy_state
        for name, module in (("pyoomph", package), ("pyoomph.output", output_package),
                             ("pyoomph.output.states", states)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(paper_bubble, "build_problem", lambda *args, **kwargs: ProblemStandin())
        monkeypatch.setattr(paper_bubble, "_stationary_residual", lambda problem: 0.0)
        runner.create_run_manifest(tmp_path, {"purpose": "phase velocity software fixture only"})
        return paper_bubble.run_model(case, tmp_path)

    return execute


@pytest.mark.parametrize("liquid,gas,frame,failed_check", [
    (3e-9, 4e-9, -5e-9, None),
    (2e-8, 4e-9, -5e-9, "liquid_static_velocity"),
    (3e-9, 2e-8, -5e-9, "gas_static_velocity"),
    (3e-9, 4e-9, -2e-8, "static_frame_speed"),
])
def test_static_summary_records_rms_and_rejects_nonzero_phase_or_frame_velocity(
    phase_velocity_summary_standin, liquid, gas, frame, failed_check,
):
    summary = phase_velocity_summary_standin("verification/paper_model/cases/static-laplace.json",
                                             liquid, gas, frame)
    assert summary["liquid_velocity_rms"] == pytest.approx(liquid, abs=1e-20)
    assert summary["gas_velocity_rms"] == pytest.approx(gas, abs=1e-20)
    assert summary["frame_speed"] == frame
    assert summary["completion_reached"] is (failed_check is None)
    assert summary["scientific_status"] == ("checks_passed" if failed_check is None else "checks_failed")
    if failed_check:
        assert summary["checks"][failed_check] is False
        assert all(value for name, value in summary["checks"].items() if name != failed_check)


@pytest.mark.parametrize("case_path,has_gas", [
    ("verification/paper_model/cases/bare-tube.json", False),
    ("validation/herradaEtAl_paper_model/cases/h15-bo1p56.json", True),
])
def test_translating_and_bare_cases_record_rms_without_zero_flow_gate(
    phase_velocity_summary_standin, case_path, has_gas,
):
    summary = phase_velocity_summary_standin(case_path, 0.03, 0.02, -0.01)
    assert summary["liquid_velocity_rms"] == pytest.approx(0.03)
    assert summary["gas_velocity_rms"] == (pytest.approx(0.02) if has_gas else None)
    assert not {"liquid_static_velocity", "gas_static_velocity", "static_frame_speed"} & set(summary["checks"])
    assert summary["completion_reached"] is True
