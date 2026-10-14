"""Marker-only metadata and source-owned empty-object route.

The implemented source interface needs an exact root phase and official
active admission. Native imports occur only in the admitted executor.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from datetime import UTC, datetime
from pathlib import Path

from runners import paper_dynamic_callback_protocol as protocol

require = protocol.require
SCHEMA = protocol.SCHEMA
PYTHON_SOURCE_HASHES = {
    "pyoomph/__init__.py": "1aa7d956d78da567084566bcd4acf34c340d9900107ead4bd9fc650df13c4b7a",
    "pyoomph/generic/problem.py": "b75d6a810daddb3e8f0853c60fea780b33220c6c09cf819e5f219f5af056018f",
    "pyoomph/solvers/generic.py": "dc485de4f8eeb03863754ce95598c254ce428a5a3418d5bdbff6ebf84554d771",
    "pyoomph/generic/ccompiler.py": "392c9837322c2efe5422810ec71fab1224fe78a42d4aa022b0523ecdef35916e",
}
POLICY = dict(protocol.MARKER_POLICY)
_COMPILER_POLICY_OWNERS = {}
CLAIMS = {
    "warm_cold_qualified": False, "restart_history_qualified": False,
    "full_newton_sequence_qualified": False, "scientific_acceptance": False,
    "numerical_acceptance": False, "production_jump_qualified": False,
    "nonflat_physical_history_qualified": False, "paper_agreement_qualified": False,
    "synthetic_buffer_stress_performed": False,
}


def no_scientific_before_admission():
    require(not any(name in {"numpy", "scipy", "pyoomph"} or name.startswith(
        ("numpy.", "scipy.", "pyoomph.", "_pyoomph")) for name in sys.modules),
        "scientific/native packages were imported before marker admission")


def phase_contract(phase, requirements, *, now=None):
    require(type(requirements) is dict and requirements.get("role") == "marker"
            and requirements.get("schema") == SCHEMA + "-requirements", "marker requirements required")
    protocol.caps_contract(requirements["caps"])
    protocol.controls_contract(requirements["controls"])
    require(type(phase) is dict and set(phase) == {"schema", "requirements", "issued_by", "declared_at",
            "deadline_utc", "phase_authorized"} and phase["schema"] == SCHEMA + "-phase"
            and phase["issued_by"] == "root" and phase["phase_authorized"] is True
            and protocol.same(phase["requirements"], requirements), "exact root marker phase differs")
    current = datetime.now(UTC) if now is None else now
    start, end = protocol.utc(phase["declared_at"]), protocol.utc(phase["deadline_utc"])
    require(start <= current < end and 0 < (end - start).total_seconds() <= 2700,
            "finite marker deadline expired or invalid")
    return end


def preflight(args, driver, *, component=None, observed=None, committed_reader=None):
    """This branch cannot consume a stationary dump or call parent_contract."""
    no_scientific_before_admission()
    require(args.role == "marker" and args.warm_result is None, "marker cannot consume warm/cold results")
    require(args.source.resolve(strict=True) == driver.ROOT, "selected isolated marker source required")
    driver.fidelity.no_cached_foreign_helpers(driver.ROOT)
    book = driver.Book()
    bindings_ref = driver.fidelity.reference(args.bindings.absolute())
    bindings = book.read(bindings_ref)
    require(type(bindings) is dict and set(bindings) == {"schema", "case", "environment", "inherited_time_step_charge_spent"}
            and bindings["schema"] == SCHEMA + "-bindings"
            and type(bindings["inherited_time_step_charge_spent"]) is int
            and bindings["inherited_time_step_charge_spent"] == 1, "scalar-only marker bindings required")
    protocol.reference_contract(bindings["case"])
    case = driver.model.validate_case(book.read(bindings["case"]))
    physical = case["physics"]
    require(physical["problem"] == "finite_gas_bubble" and physical["ca_in"] == -.002
            and physical["bond"] == 1.56 and physical["vhat"] == 3.3 and physical["domain_over_r"] == 15
            and case["pressure"] == {"field": "static", "reference": "lower_static_zero", "body_force": "explicit_both_phases"},
            "unchanged I scalar case/formulation required")
    pending = []
    selected = dict(protocol.MARKER_CONTROLS)
    controls_ref = None
    if args.controls is None:
        pending.append("root marker controls unissued")
    else:
        controls_ref = driver.fidelity.reference(args.controls.absolute())
        selected = protocol.controls_contract(book.read(controls_ref))
    component = driver.ordinary.component_revision(driver.ROOT) if component is None else component
    observed = driver.ordinary.pyoomph_revision() if observed is None else observed
    try:
        driver.ordinary.require_clean_source(component)
    except ValueError:
        pending.append("marker driver is not clean published source")
    source = {"commit": component["head"], "root": str(driver.ROOT), "files": driver.fidelity.source_files(driver.ROOT)}
    if component.get("clean") is True:
        for name, ref in source["files"].items():
            raw = (driver.ordinary._git(driver.ROOT, "show", "HEAD:" + name) if committed_reader is None
                   else committed_reader(name))
            require(raw is not None and hashlib.sha256(raw).hexdigest() == ref["sha256"],
                    "marker source closure is not committed: " + name)
    driver.ordinary.require_pinned_environment(observed)
    expected = bindings["environment"]
    require(type(expected) is dict and set(expected) == {"pin", "native_core", "interpreter", "receipt", "python_sources"}
            and expected["pin"] == observed["commit"] == protocol.PYOOMPH_PIN
            and protocol.same(expected["native_core"], observed["native_core"])
            and expected["interpreter"] == observed["interpreter"]["executable"], "exact pinned marker environment differs")
    protocol.reference_contract(expected["native_core"])
    book.file(expected["native_core"])
    require(type(expected["python_sources"]) is dict and set(expected["python_sources"]) == set(PYTHON_SOURCE_HASHES),
            "exact installed pinned Python source closure required")
    for name, ref in expected["python_sources"].items():
        protocol.reference_contract(ref)
        require(ref["sha256"] == PYTHON_SOURCE_HASHES[name], "installed pinned Python source differs: " + name)
        book.file(ref)
    receipt_ref = None
    if args.environment is None:
        pending.append("root pinned environment byte receipt unissued")
    else:
        receipt_ref = driver.fidelity.reference(args.environment.absolute())
        require(protocol.same(receipt_ref, expected["receipt"]), "marker environment receipt selector differs")
        receipt = book.read(receipt_ref)
        require(receipt.get("schema") == "pyoomph-environment-v1" and receipt.get("compatible") is True
                and receipt.get("checked_compiler") == "system" and receipt.get("checked_solver") == "superlu"
                and receipt["checks"]["compiler"]["status"] == receipt["checks"]["solver"]["status"] == "passed"
                and receipt["pyoomph"]["resolved_commit"] == protocol.PYOOMPH_PIN
                and receipt["pyoomph"]["core_sha256"] == observed["native_core"]["sha256"]
                and receipt["interpreter"]["path"] == observed["interpreter"]["executable"]
                and receipt["mpi"]["compiled"] is False, "root serial pinned environment byte evidence differs")
    if not protocol.MARKER_DRIVER_INTERFACE_SUPPORTED:
        pending.append("marker official operations/interface review incomplete; native execution remains refused")
    output = None if args.out is None else args.out.absolute()
    if output is None:
        pending.append("marker output unissued")
    else:
        require(not any(p.is_symlink() for p in (output, *output.parents)), "marker output alias refused")
        for ref in [*book.records.values(), *source["files"].values()]:
            path = Path(ref["path"])
            require(not path.is_relative_to(output) and not output.is_relative_to(path.parent)
                    and not output.is_relative_to(driver.ROOT), "marker output overlaps source/input authority")
    requirements = {"schema": SCHEMA + "-requirements", "role": "marker", "source_pin": protocol.PYOOMPH_PIN,
        "bindings": bindings_ref, "source": source, "environment": observed, "fresh_environment": receipt_ref,
        "controls": selected, "controls_ref": controls_ref, "caps": dict(protocol.MARKER_CAPS), "case": case,
        "case_ref": bindings["case"], "output": None if output is None else str(output), "warm_result": None,
        "inputs": sorted(book.records.values(), key=lambda row: row["path"]), "policy": dict(POLICY),
        "inherited_time_step_charge_spent": 1, "new_time_step_allowance": 0,
        "callback_protocol": {"schema": SCHEMA, "type_forwarding_before_construction": True,
            "same_instance_dispatch_only": True, "marker_interface_supported": protocol.MARKER_DRIVER_INTERFACE_SUPPORTED,
            "full_fidelity_interface_supported": False}}
    phase = phase_ref = None
    if args.phase is None:
        pending.append("root marker phase declaration unissued")
    else:
        phase_ref = driver.fidelity.reference(args.phase.absolute())
        phase = book.read(phase_ref)
        phase_contract(phase, requirements)
    return {"case": case, "case_ref": bindings["case"], "bindings": bindings, "bindings_ref": bindings_ref,
            "book": book, "controls": selected, "component": component, "source": source, "environment": observed,
            "output": output, "phase": phase, "phase_ref": phase_ref, "requirements": requirements,
            "pending": pending, "native_execution_authorized": False}


def bind_authority(prepared, driver):
    def refresh():
        prepared["book"].stable()
        admission = prepared["book"].read(prepared["admission_ref"])
        actual = driver.admission_contract(admission, prepared)
        return {"requirements": prepared["requirements"], "phase": prepared["phase"],
                "phase_ref": prepared["phase_ref"], "admission": admission,
                "admission_ref": prepared["admission_ref"], "native_process": actual}
    return protocol.bind_admitted_marker_authority(refresh(), refresh)


def admission_header_contract(admission, prepared):
    require(type(admission) is dict and set(admission) == {"schema", "native_execution_authorized", "marker_authorized",
            "phase", "requirements", "active_tracker", "role", "deadline_utc"}
            and admission["schema"] == SCHEMA + "-admission" and admission["role"] == "marker"
            and admission["native_execution_authorized"] is True and admission["marker_authorized"] is True
            and protocol.same(admission["requirements"], prepared["requirements"])
            and protocol.same(admission["phase"], prepared["phase_ref"])
            and admission["deadline_utc"] == prepared["phase"]["deadline_utc"],
            "complete exact official marker admission header required")
    protocol.reference_contract(admission["active_tracker"])


class _ConstructorPolicy:
    """Count only the allowed availability pass; prohibit FEM compiler entry."""
    def __init__(self, owner, base_compiler, system_compiler, deny_names, record):
        self.owner, self.base, self.system, self._record = owner, base_compiler, system_compiler, record
        self.passes, self.denied, self.patches = 0, [], []
        self.backend_entries = self.backend_returns = 0
        self.errors = []
        self.acquired = False
        self.selected_compiler = self.linear_solver_default = None
        self.verified = False
        self.availability = vars(base_compiler)["available_compilers"]
        require(isinstance(self.availability, classmethod), "exact source-bound availability classmethod required")
        self.deny_names = deny_names

    def fail(self, stage, error):
        self.verified = False
        item = {"stage": stage, "type": type(error).__name__, "message": str(error),
                "constructor_compiler_availability_passes": self.passes,
                "constructor_compiler_backend_entries": self.backend_entries,
                "constructor_compiler_backend_returns": self.backend_returns}
        if item not in self.errors:
            self.errors.append(item)

    def record(self, label, value):
        try:
            return self._record(label, value)
        except BaseException as error:  # A caught recorder failure must remain sticky.
            self.fail("record:" + label, error)
            raise

    def require_healthy(self):
        require(not self.errors and not self.denied, "constructor policy has a sticky failure")
        require(self.backend_entries == self.backend_returns == self.passes,
                "availability attempt cannot qualify missing or failed backend work")

    def install(self):
        self.owner.require_live_owner()
        protocol.require_authority(self.owner.authority)
        require(self.base not in _COMPILER_POLICY_OWNERS, "exact compiler class already owned")
        original = self.availability.__func__

        def availability(cls):
            try:
                self.owner.require_live_owner()
                protocol.require_authority(self.owner.authority)
                self.require_healthy()
                if self.passes >= 1:
                    self.denied.append("constructor_compiler_availability")
                    self.record("forbidden_marker_native_entry", {"api": "constructor_compiler_availability"})
                    raise ValueError("second compiler availability pass refused")
                self.passes += 1
                self.record("constructor_compiler_availability_pass", {"constructor_compiler_availability_passes": self.passes})
                self.backend_entries += 1
                result = original(cls)
                self.backend_returns += 1
                self.record("constructor_compiler_availability_returned", {
                    "constructor_compiler_availability_passes": self.passes,
                    "constructor_compiler_backend_entries": self.backend_entries,
                    "constructor_compiler_backend_returns": self.backend_returns})
                protocol.require_authority(self.owner.authority)
                self.require_healthy()
                return result
            except BaseException as error:  # Pinned getter swallows Exception; retain failed backend/record evidence.
                self.fail("compiler_availability", error)
                raise

        pending = [(self.base, "available_compilers", classmethod(availability))]
        for cls, names in [(self.owner.problem_type, self.deny_names), (self.base, {"compile"})]:
            for name in sorted(names):
                if not hasattr(cls, name):
                    continue
                def denied(*args, _name=name, **kwargs):
                    self.denied.append(_name)
                    self.record("forbidden_marker_native_entry", {"api": _name})
                    raise ValueError("marker-only role prohibits " + _name)
                pending.append((cls, name, denied))
        # Capture and verify every descriptor before any mutation.
        for cls, name, shim in pending:
            own = vars(cls).get(name, protocol._MISSING)
            inherited = inspect.getattr_static(cls, name)
            self.patches.append((cls, name, own, inherited, shim))
        require(all(vars(cls).get(name, protocol._MISSING) is own and inspect.getattr_static(cls, name) is inherited
                    for cls, name, own, inherited, shim in self.patches), "foreign constructor policy descriptor changed")
        _COMPILER_POLICY_OWNERS[self.base] = self
        self.acquired = True
        for cls, name, own, inherited, shim in self.patches:
            setattr(cls, name, shim)

    def selected(self, problem):
        try:
            self.require_installed()
            self.require_healthy()
            compiler = problem.get_ccompiler()
            require(type(compiler) is self.system and compiler.compiler_id == "system"
                    and self.system.__module__ == "pyoomph.generic.ccompiler" and self.system.__name__ == "SystemCCompiler",
                    "unexpected selected compiler refused")
            require(type(problem._lasolver) is str and problem._lasolver == "superlu", "constructed scalar linear default differs")
            require(problem._initialised is False, "marker constructor initialised equations")
            self.selected_compiler = {"module": self.system.__module__, "class": self.system.__name__, "compiler_id": compiler.compiler_id}
            self.linear_solver_default = "superlu"
            self.record("constructor_policy_verified", {"selected_compiler": self.selected_compiler,
                        "linear_solver_default": "superlu", "constructor_policy_verified": True,
                        "constructor_compiler_availability_passes": self.passes, "fem_equation_compilations": 0})
            self.require_healthy()
            self.verified = True
        except BaseException as error:  # Caught verification/record failure must prevent later charge.
            self.fail("verify_instance", error)
            raise

    def require_marker_gate(self):
        self.owner.require_live_owner()
        protocol.require_authority(self.owner.authority)
        self.require_installed()
        self.require_healthy()
        require(self.verified is True, "constructor policy verification is absent")

    def require_installed(self):
        require(self.acquired and _COMPILER_POLICY_OWNERS.get(self.base) is self
                and all(vars(cls).get(name, protocol._MISSING) is shim
                        for cls, name, own, inherited, shim in self.patches),
                "exact constructor policy ownership/descriptor binding changed")

    def close(self):
        if not self.acquired:
            return
        self.require_installed()
        if self.owner.construction_attempted:
            self.record("constructor_policy_retained", {"shim_retained_until_process_exit": True})
            return
        require(all(vars(cls).get(name, protocol._MISSING) is shim for cls, name, own, inherited, shim in self.patches),
                "foreign constructor policy descriptor changed; no restoration mutation")
        for cls, name, own, inherited, shim in reversed(self.patches):
            if own is protocol._MISSING:
                delattr(cls, name)
            else:
                setattr(cls, name, own)
        del _COMPILER_POLICY_OWNERS[self.base]


def execute(prepared, output, emit, driver):
    require(protocol.MARKER_DRIVER_INTERFACE_SUPPORTED is True,
            "marker interface remains unsupported pending full official wiring/review")
    require(prepared.get("native_execution_authorized") is True and not prepared["pending"]
            and prepared["requirements"]["role"] == "marker", "exact admitted marker-only entry required")
    no_scientific_before_admission()
    authority = prepared["marker_authority"]
    protocol.require_authority(authority)
    # These imports are unreachable before the exact root phase/active gate.
    from pyoomph import _pyoomph_core as core
    prepared["native_imported"] = True
    from pyoomph.generic.ccompiler import BaseCCompiler, SystemCCompiler
    from pyoomph.solvers import generic as solver_defaults

    require(protocol.same(driver.fidelity.reference(Path(core.__file__).absolute()),
                          prepared["environment"]["native_core"]), "imported native core bytes differ")
    expected_sources = prepared["bindings"]["environment"]["python_sources"]
    for name, module_name in {"pyoomph/__init__.py": "pyoomph", "pyoomph/generic/problem.py": "pyoomph.generic.problem",
            "pyoomph/solvers/generic.py": "pyoomph.solvers.generic", "pyoomph/generic/ccompiler.py": "pyoomph.generic.ccompiler"}.items():
        require(protocol.same(driver.fidelity.reference(Path(sys.modules[module_name].__file__).absolute()), expected_sources[name]),
                "actual imported pinned Python module differs: " + name)
    require(driver.model._native_problem_types.cache_info().currsize == 0, "fresh isolated native factory cache required")
    problem_type = driver.model._native_problem_types()[0]
    descriptor = vars(core.Problem)[protocol.HOOK]
    require(core.Problem in problem_type.__mro__ and inspect.getattr_static(problem_type, protocol.HOOK) is descriptor,
            "source-owned fresh Problem type/native descriptor differs")
    return _run_declared_marker(prepared, output, emit, driver, core, problem_type,
                                BaseCCompiler, SystemCCompiler, solver_defaults)


def _run_declared_marker(prepared, output, emit, driver, core, problem_type,
                         BaseCCompiler, SystemCCompiler, solver_defaults):
    """Source-owned body after the executor bound actual core/package/class bytes.

    Supplied class shapes in software controls are explicitly synthetic. This
    helper does not establish their native origin or grant entry authority.
    """
    authority = prepared["marker_authority"]
    protocol.require_authority(authority)
    descriptor = vars(core.Problem)[protocol.HOOK]
    require(core.Problem in problem_type.__mro__ and inspect.getattr_static(problem_type, protocol.HOOK) is descriptor,
            "declared native class/descriptor differs")
    charges = protocol.DurableMarkerCharges(output, authority, driver.fidelity.write_new, driver.fidelity.read_bound)
    owner = protocol.BeforeStepTypeForwarding(problem_type, authority, charges, emit)
    prepared["marker_owner"], prepared["marker_charges"] = owner, charges
    forbidden = set(driver.fidelity.EvolutionGuard.REQUIRED) | {
        "define_problem", "ndof", "get_mesh", "get_ode", "get_current_dofs", "get_history_dofs", "get_local_dofs",
        "get_current_pinned_values", "set_current_dofs", "_get_all_values_at_current_time", "_set_all_values_at_current_time",
        "get_residuals", "_assemble_residual_jacobian", "_assemble_hessian_tensor", "save_state", "output",
        "_define_state_file", "assign_eqn_numbers", "setup_for_stability_analysis",
    }
    policy = _ConstructorPolicy(owner, BaseCCompiler, SystemCCompiler, forbidden, emit)
    owner.record = policy.record
    record = policy.record
    prepared["constructor_policy"] = policy
    problem = None
    try:
        with owner:
            try:
                policy.install()
                protocol.require_authority(authority)
                solver_defaults.set_default_linear_solver("superlu")
                require(solver_defaults.get_default_linear_solver() == "superlu", "scalar solver default differs")
                problem = owner.construct_case(prepared["case"])
                policy.selected(problem)
                policy.require_marker_gate()
                readiness = protocol.qualify_cpp_before_step_marker(problem, owner, core.Problem, descriptor, record)
                dispatch = protocol.require_dispatch_marker_readiness(problem, readiness)
                evidence = {"schema": SCHEMA + "-evidence", "role": "marker", "authority": authority.snapshot(),
                    "construction_charge": charges.references["construction"], "marker_charge": charges.references["marker"],
                    "local_identity": {"owner_id": id(owner), "problem_type_id": id(problem_type),
                        "instance_id": id(problem), "descriptor_id": id(descriptor)}, "raw_marker_deliveries": 1,
                    "normal_descriptor_return": True, **dispatch, "readiness_transferable": False,
                    "positive_ndof_observed": None, "selected_compiler": policy.selected_compiler,
                    "linear_solver_default": policy.linear_solver_default, "constructor_policy_verified": policy.verified, **CLAIMS}
                prepared["marker_evidence"] = driver.fidelity.write_new(output / "marker-evidence.json", evidence)
                protocol.require_authority(authority)
            except BaseException as error:  # Body/original callback failure stays sticky if an outer caller catches it.
                policy.fail("marker_body", error)
                raise
            finally:
                if problem is None:
                    known = [value for value in owner.live.values() if type(value) is problem_type]
                    if len(known) == 1:
                        problem = known[0]
                if problem is not None:
                    try:
                        owner.release_instance(problem)
                    except BaseException as error:  # Retain original release/callback failure without retry.
                        policy.fail("release", error)
                        raise
                    finally:
                        prepared["release_returned"] = id(problem) in owner.release_completed
                policy.close()
    finally:
        prepared["shim_retained_until_process_exit"] = (owner.installed and owner.construction_attempted
            and protocol._CLASS_OWNERS.get(problem_type) is owner and vars(problem_type).get(protocol.HOOK) is owner.shim)
    require(prepared.get("release_returned") is True and prepared["shim_retained_until_process_exit"] is True,
            "explicit release/terminal-retained ownership missing")
    protocol.require_authority(authority)
    policy.require_healthy()
    return {"normal_descriptor_return": True, "same_instance_dispatch": True, "release_returned": True,
            "shim_retained_until_process_exit": True, "marker_evidence": prepared["marker_evidence"],
            "constructor_compiler_availability_passes": policy.passes, "selected_compiler": policy.selected_compiler,
            "linear_solver_default": policy.linear_solver_default, "constructor_policy_verified": policy.verified}


def initial_result(prepared):
    return {"schema": SCHEMA + "-result", "role": "marker", "status": "failed_marker",
            "requirements": prepared["requirements"], "source": prepared["source"], "controls": prepared["controls"],
            "caps": dict(protocol.MARKER_CAPS), "phase": prepared["phase_ref"], "admission": None,
            "native_process": None, "native_imported": False, "problem_constructions_charged": 0,
            "cpp_virtual_marker_calls_charged": 0, "constructor_compiler_availability_passes": 0,
            "raw_marker_deliveries": 0, "normal_descriptor_return": False, "same_instance_dispatch": False,
            "release_returned": False, "shim_retained_until_process_exit": False,
            "selected_compiler": None, "linear_solver_default": None, "constructor_policy_verified": False,
            "constructor_compiler_backend_entries": 0, "constructor_compiler_backend_returns": 0,
            "constructor_policy_failed": False, "constructor_policy_errors": [],
            "native_destruction_proved": False, "native_process_terminal_proved": False,
            "construction_charge": None, "marker_charge": None, "marker_evidence": None,
            "raw_events": [], "forbidden_native_entry_attempts": [], "time_step_calls_charged": 0,
            "inherited_time_step_charge_spent": 1, "new_time_step_allowance": 0, **CLAIMS}


def retain_result_state(result, prepared):
    result["native_imported"] = prepared.get("native_imported", False)
    result["admission"] = prepared.get("admission_ref")
    result["native_process"] = prepared.get("native_process")
    result["raw_events"] = list(prepared.get("raw_events", []))
    result["raw_marker_deliveries"] = prepared.get("raw_marker_deliveries", 0)
    result["normal_descriptor_return"] = prepared.get("normal_descriptor_return", False)
    result["release_returned"] = prepared.get("release_returned", False)
    result["shim_retained_until_process_exit"] = prepared.get("shim_retained_until_process_exit", False)
    owner = prepared.get("marker_owner")
    if owner is not None:
        result["raw_marker_deliveries"] = owner.raw_marker_deliveries
        result["normal_descriptor_return"] = owner.normal_descriptor_return
    charges = prepared.get("marker_charges")
    if charges is not None:
        result["problem_constructions_charged"] = int("construction" in charges.attempted)
        result["cpp_virtual_marker_calls_charged"] = int("marker" in charges.attempted)
        result["construction_charge"] = charges.references.get("construction")
        result["marker_charge"] = charges.references.get("marker")
    result["marker_evidence"] = prepared.get("marker_evidence")
    policy = prepared.get("constructor_policy")
    if policy is not None:
        result["constructor_compiler_availability_passes"] = policy.passes
        result["forbidden_native_entry_attempts"] = list(policy.denied)
        result["selected_compiler"] = policy.selected_compiler
        result["linear_solver_default"] = policy.linear_solver_default
        result["constructor_policy_verified"] = policy.verified
        result["constructor_compiler_backend_entries"] = policy.backend_entries
        result["constructor_compiler_backend_returns"] = policy.backend_returns
        result["constructor_policy_failed"] = bool(policy.errors or policy.denied)
        result["constructor_policy_errors"] = [dict(value) for value in policy.errors]
