"""Strict software contracts for one unchanged-case capture fidelity step.

Native packages are caller-supplied only after separate phase admission.
Software checks and synthetic fixtures cannot establish native fidelity.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from runners import paper_dynamic_capture as capture_module
from runners import paper_dynamic_jump as study
from runners import paper_dynamic_jump_runtime as runtime
from runners import paper_dynamic_state as numeric_state
from runners import run_paper_cold_history as history
from runners import run_paper_finite_gas_fold as fold

require = study.require
SCHEMA = "stone-dynamic-fidelity-v1"
ROLES = ("warm", "cold")
CAPS = {"time_step_calls": 1, "stationary_solves": 0, "eigensolves": 0,
        "remeshes": 0, "refinements": 0, "inversion_retries": 0}
CONTROL_KEYS = {"dt_solver", "max_newton_iterations", "max_newton_updates",
                "newton_tolerance"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "absolute regular authority required")
    return {"path": str(path.resolve(strict=True)), "sha256": digest(path)}


def bound(record):
    require(isinstance(record, dict) and {"path", "sha256"} <= set(record), "bound artifact required")
    actual = reference(record["path"])
    require(actual["sha256"] == record["sha256"], "artifact bytes changed: " + actual["path"])
    return Path(actual["path"])


def read_bound(record):
    """Return the same unchanged file bytes whose authority hash was checked."""
    require(isinstance(record, dict) and {"path", "sha256"} <= set(record), "bound artifact required")
    path = Path(record["path"])
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
            "absolute regular authority required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "regular opened authority required")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    current = path.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, key) == getattr(after, key) == getattr(current, key) for key in keys)
            and len(raw) == before.st_size, "authority changed during read")
    actual = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    require(actual["sha256"] == record["sha256"], "artifact bytes changed: " + str(path))
    return path, raw, actual


def write_new(path, value):
    """Exclusive durable receipt; never replace an earlier charge or failure."""
    path = Path(path)
    require(path.is_absolute() and path.parent.is_dir()
            and not any(p.is_symlink() for p in (path.parent, *path.parent.parents)), "regular receipt directory required")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


def controls(value, *, unissued=False):
    require(isinstance(value, dict) and set(value) == CONTROL_KEYS, "exact fidelity controls required")
    missing = [key for key in sorted(CONTROL_KEYS) if value[key] is None]
    require(unissued or not missing, "fidelity numerical controls are unissued")
    for key in CONTROL_KEYS - {"max_newton_iterations", "max_newton_updates"}:
        if value[key] is not None:
            study.number(value[key], key, positive=True)
    for key in ("max_newton_iterations", "max_newton_updates"):
        if value[key] is not None:
            study.integer(value[key], key)
    if not missing:
        require(value["max_newton_updates"] <= value["max_newton_iterations"],
                "update cap exceeds the one-call iteration cap")
    return copy.deepcopy(value), missing


def source_files(root):
    root = Path(root)
    names = sorted({str(path.relative_to(root)) for directory in ("problems", "runners")
                    for path in (root / directory).glob("*.py")} | {"pyproject.toml", "uv.lock"})
    return {name: reference(root / name) for name in names}


def no_cached_foreign_helpers(root):
    import sys
    for name, module in list(sys.modules.items()):
        if name.startswith(("runners.", "problems.")):
            path = getattr(module, "__file__", None)
            require(path is not None and Path(path).resolve(strict=True)
                    == Path(root) / (name.replace(".", "/") + ".py"), "foreign cached helper: " + name)


class LoadAdapter:
    """One administrative load with an exact owned template/remesher graph.

    At the pinned loader an equal relative spelling avoids template replacement.
    Content equality proves when that spelling can select the existing live mesh.
    A fresh differing mesh may be reconstructed once from its bound bytes.
    """

    def __init__(self, problem, capture, guard, record):
        self.problem, self.capture, self.guard, self.record = problem, capture, guard, record
        self.calls = 0

    def load(self, state_ref, meshes):
        require(self.calls == 0, "an additional native load requires a new declared phase")
        state_path = bound(state_ref)
        parsed = history.serialized_metadata(state_path.read_bytes())
        refs = [name for _, _, name in parsed["meshes"]]
        require(len(refs) == len(meshes) == 1 and meshes[0]["stored_path"] == refs[0],
                "fidelity load requires its one exact mesh reference")
        target = bound(meshes[0])
        require(target == (state_path.parent / refs[0]).resolve(strict=True), "state mesh reference differs")
        original, active = runtime.owned_quality_templates(self.problem, self.capture)
        owner = active.remesher
        old = fold.template_registration(self.problem)
        live = reference(Path(active._meshfile).absolute())
        preserve = live["sha256"] == meshes[0]["sha256"]
        selected = Path(live["path"]) if preserve else target
        exchange = {refs[0]: os.path.relpath(selected, state_path.parent)}
        self.calls += 1  # An interrupted load cannot be retried inside this role.
        self.record("native_load_charged", {"load_calls": self.calls, "state": state_ref,
                    "administrative_reconstruction_permitted": not preserve, "registration_before": old})
        with self.guard.administrative_load():
            self.problem.load_state(str(state_path), quiet=True, additional_info={"exchange_msh_file": exchange})
        new_original, new_active = runtime.owned_quality_templates(self.problem, self.capture)
        require(new_original is original and new_active.remesher is owner,
                "native load replaced the original template or owned remesher")
        require(not preserve or new_active is active, "equal-byte canonical load replaced its active template")
        require(reference(Path(new_active._meshfile).absolute())["sha256"] == meshes[0]["sha256"],
                "loaded active template has unbound mesh bytes")
        require({name: float(self.problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")}
                == {name: parsed_value for name, parsed_value in self.guard.forcing.items()},
                "native load changed the declared forcing")
        self.record("native_load_returned", {"load_calls": self.calls, "registration_after": fold.template_registration(self.problem),
                    "existing_active_template_preserved": preserve})
        return parsed


class EvolutionGuard:
    """Refuse undeclared solver/remesh entrypoints before native initialisation."""

    ALWAYS = frozenset({"solve_stationary", "solve_eigenproblem", "run", "run_with_adaptive_timestepping",
              "arclength_continuation", "activate_bifurcation_tracking", "force_remesh", "adapt",
              "actions_before_remeshing"})
    STEP_ONLY = frozenset({"solve", "solve_unsteady", "unsteady_newton_solve", "actions_before_newton_solve"})
    FILL = frozenset({"assign_initial_values_impulsive", "assign_initial_positions_impulsive"})

    def __init__(self, problem, forcing, record):
        self.problem, self.forcing, self.record = problem, forcing, record
        self.scope = None
        self.denied = []
        self.callbacks = []
        self.step_entries = {}
        self.restored = []
        self.initialisations = self.loads = self.steps = 0

    def wrapper(self, name, original):
        def call(*args, **kwargs):
            permitted = ((name in self.STEP_ONLY and self.scope == "step")
                         or (name in self.FILL and self.scope == "initialise")
                         or (name == "load_state" and self.scope == "load")
                         or (name == "actions_after_remeshing" and self.scope == "load"))
            if not permitted:
                self.denied.append(name)
                self.record("forbidden_native_entry", {"api": name, "scope": self.scope})
                raise ValueError("fidelity role prohibits " + name)
            if name in {"solve", "solve_unsteady", "unsteady_newton_solve"}:
                require(self.step_entries.get(name, 0) == 0, "additional native time-step entry refused: " + name)
                self.step_entries[name] = 1
            self.callbacks.append({"api": name, "scope": self.scope})
            return original(*args, **kwargs)
        return call

    @contextlib.contextmanager
    def installed(self):
        names = self.ALWAYS | self.STEP_ONLY | self.FILL | {"actions_after_remeshing", "load_state"}
        try:
            for name in sorted(names):
                original = getattr(self.problem, name, None)
                if not callable(original):
                    require(name not in {"solve", "solve_stationary", "solve_eigenproblem", "force_remesh", "adapt",
                                         "actions_before_newton_solve", "actions_before_remeshing", "actions_after_remeshing"},
                            "required guarded API absent: " + name)
                    continue
                present, stored = name in vars(self.problem), vars(self.problem).get(name)
                wrapped = self.wrapper(name, original)
                setattr(self.problem, name, wrapped)
                self.restored.append((name, present, stored, wrapped))
            yield self
            require(not self.denied, "fidelity role attempted forbidden native work")
        finally:
            unchanged = all(getattr(self.problem, name) is wrapped for name, _, _, wrapped in self.restored)
            for name, present, stored, _ in reversed(self.restored):
                if present:
                    setattr(self.problem, name, stored)
                else:
                    delattr(self.problem, name)
            self.restored.clear()
            require(unchanged, "guard callback inventory changed")

    @contextlib.contextmanager
    def declared(self, name):
        require(self.scope is None, "nested fidelity mutation scope")
        attribute = {"initialise": "initialisations", "load": "loads", "step": "steps"}[name]
        require(getattr(self, attribute) == 0, "fidelity operation allowance exhausted: " + name)
        setattr(self, attribute, 1)
        self.scope = name
        try:
            yield
        finally:
            self.scope = None

    def administrative_load(self):
        return self.declared("load")


def endtime_restore(problem, source_bytes, case, output, record):
    """Restore only the pinned loader/writer scalar asymmetry and prove its bytes."""
    before_path = output / "loaded-before-endtime.dump"
    problem.save_state(str(before_path), quiet=True)
    before = before_path.read_bytes()
    restored = history.restore_endtime(problem, source_bytes, before, case)
    after_path = output / "loaded-after-endtime.dump"
    problem.save_state(str(after_path), quiet=True)
    after = after_path.read_bytes()
    a, b = history.serialized_metadata(before), history.serialized_metadata(after)
    offset = a["endtime_offset"]
    require(b["endtime_offset"] == offset and before[:offset] == after[:offset]
            and before[offset + 8:] == after[offset + 8:] and before[offset:offset + 8] != after[offset:offset + 8],
            "load metadata restoration changed bytes outside the endtime scalar")
    require(b["endtime"] == restored, "restored writer endtime differs")
    record("endtime_metadata_restored", {"value": restored, "only_changed_slot": [offset, offset + 8],
                "before": reference(before_path), "after": reference(after_path)})
    return after


def raw_profile(capture, problem):
    witness = capture(problem)
    return {"schema": SCHEMA + "-raw-profile", "source": "raw current interface geometry; no interpolation",
            "domains": {name: {key: copy.deepcopy(view[key]) for key in ("nodes", "connectivity", "stride")}
                        for name, view in witness["geometry"]["domains"].items() if name.endswith("/interface")}}


def cross_check_live(problem, capture, data, record, label):
    before = capture(problem)
    checked = capture_module.cross_check(numeric_state.decode(data), before)
    require(capture(problem) == before, "capture or numeric check changed native state")
    record(label, {"cross_check": checked, "witness": before})
    return before


def register_policy(problem, capture, record):
    before = capture(problem)
    runtime.activate_unsteady(problem, capture_module_hooks(capture))
    active = capture(problem)
    require(runtime.physical(before) == runtime.physical(active), "activation changed complete physical history")
    for name, value in runtime.DYNAMIC_FLAGS.items():
        setattr(problem, name, value)
    flagged = capture(problem)
    require(runtime.physical(active, omit=("flags",)) == runtime.physical(flagged, omit=("flags",)),
            "continuation flags changed physical history")
    runtime.prepare_transient(problem, capture_module_hooks(capture))
    after = capture(problem)
    require(after["time"] == before["time"] and after["counter"] == before["counter"]
            and after["dts"] == before["dts"] and after["flags"] == runtime.DYNAMIC_FLAGS,
            "transient policy reset the parent clock or counter")
    record("explicit_unchanged_history_policy", {"before": before, "after": after})
    return after


def capture_module_hooks(capture):
    from types import SimpleNamespace
    return SimpleNamespace(capture=capture, template_type=capture.template_type, remesher_type=capture.remesher_type)


def retain(problem, capture, output, index, witness):
    root = output / "retained"
    root.mkdir(exist_ok=True)
    writer = capture_module.Retain(capture, root, lambda p: raw_profile(capture, p))
    row = {"accepted_step_index": index, "classification": "unchanged-case native fidelity only",
           "native_time": witness["time"], "counter": witness["counter"],
           "next_output_index": witness["policy"]["serialization"]["output_step"]}
    return writer(problem, row, witness)


def next_transition(before, after, selected):
    dt = selected["dt_solver"]
    require(after["time"] == before["time"] + dt and after["counter"] == before["counter"] + 1,
            "native next clock/counter differs")
    require(after["dts"] == [dt, *before["dts"][:-1]], "native dt history was flattened or reset")
    require(after["parameters"] == before["parameters"] and after["policy"]["all_parameters"] == before["policy"]["all_parameters"]
            and after["census"] == before["census"] and after["layout"] == before["layout"]
            and after["mode"] == before["mode"] and after["flags"] == runtime.DYNAMIC_FLAGS
            and all(s["is_steady"] is False for s in after["steppers"]), "native stepping policy or ownership drifted")
    require(runtime.stepper_registry(before) == runtime.stepper_registry(after), "native stepper registry changed")
    require(after["geometry"]["connectivity"] == before["geometry"]["connectivity"], "undeclared topology change")


def sole_step(problem, capture, guard, selected, deadline, output, record, stable):
    selected, _ = controls(selected)
    stable()
    before = capture(problem)
    require(before["flags"] == runtime.DYNAMIC_FLAGS and all(s["is_steady"] is False for s in before["steppers"]),
            "one-step fidelity must continue the saved history without startup")
    charge = write_new(output / "step-charge.json", {"schema": SCHEMA + "-step-charge", "time_step_calls_charged": 1,
                 "before_counter": before["counter"], "before_time": before["time"], "controls": selected,
                 "retries": 0, "resume_permitted": False})
    record("sole_step_charged", {"charge": charge})
    work = runtime.empty_work()
    caps = {"max_newton_iterations_per_step": selected["max_newton_iterations"],
            "max_total_newton_updates": selected["max_newton_updates"],
            "remesh": {"max_inversion_retries_per_step": 0}}
    budget = runtime.UpdateBudget(work, caps, deadline)
    bridge = runtime.NewtonCallbackBridge(problem, budget, record)
    try:
        budget.check()
        with guard.declared("step"), bridge.installed():
            problem.solve(timestep=selected["dt_solver"], temporal_error=None, spatial_adapt=0,
                          shift_values=True, do_not_set_IC=True,
                          max_newton_iterations=selected["max_newton_iterations"],
                          newton_solver_tolerance=selected["newton_tolerance"])
        budget.finish()
        work["solve_calls"] += 1
        after = capture(problem)
        next_transition(before, after, selected)
        stable()
        problem.output()
        written = capture(problem)
        runtime.output_transition(after, written)
        row = retain(problem, capture, output, 1, written)
        require(capture(problem) == written, "retention changed the complete next-transition witness")
        record("sole_step_returned", {"work": work, "callback_receipt": budget.callback_receipt(), "row": row})
        return {"initial_witness": before, "next_witness": written, "row": row, "charge": charge,
                "work": work, "callbacks": budget.callback_receipt()}
    finally:
        write_new(output / "step-work.json", {"schema": SCHEMA + "-step-work", "time_step_calls_charged": 1,
                  "work": work, "callbacks": budget.callback_receipt(), "resumable": False})
