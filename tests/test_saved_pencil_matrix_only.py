"""Closed subtype admission and truthful saved-pencil completion contracts."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def load_source(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search = load_source("runners/saved_pencil/saved_search.py", "saved_pencil_search_contract")
sys.meta_path.remove(search.DENY)
admission = load_source("runners/saved_pencil/admission.py", "saved_pencil_admission_contract")
wrapper = load_source("runners/run_bound_worker.py", "saved_pencil_wrapper_contract")


def policy(*, coarse=True):
    roles = admission.BASE_ROLES if coarse else admission.BASE_ROLES | admission.GEOMETRY_ROLES
    return {
        "schema": admission.MATRIX_ONLY_SCHEMA if coarse else admission.SCHEMA,
        "authority": {"source_commit": admission.COARSE_SOURCE if coarse else admission.SOURCE,
            "pyoomph_commit": admission.PYOOMPH, "native_core_sha256": admission.CORE,
            "strategy_sha256": "a" * 64 if coarse else admission.STRATEGY,
            "pencil_sign": "J=-dR/dU; Jv=lambda Mv"},
        "environment": {}, "inputs": {name: {} for name in roles},
        "source_closure": {"numerics.py": dict(admission.NUMERICS_ORIGIN)},
        "model": {"physics": {"problem": "finite_gas_bubble", "reynolds": 0.0,
            "bond": 1.56, "ca_in": -0.0038 if coarse else -0.00315, "r0": 1.0,
            "r_tube": admission.FACTOR, "vhat": 3.3, "domain_over_r": 15.0,
            "upper_tip_over_r": 10.0, "gas_density_ratio": 0.000949,
            "gas_viscosity_ratio": 1.8e-5, "seed_radius_over_tube": 0.8}},
        "target": {"n": 316989, "j_nnz": 10490006 if coarse else 10488602,
            "m_nnz": 11836, "mass_rows": 1481, "mass_columns": 2960,
            "mass_rows_sha256": "b" * 64,
            "globals": {"globals/U": [316987], "gas_volume/pressure": [316988]},
            "forced": [], "m": 0, "time_factor": admission.FACTOR},
        "gates": dict(admission.LIMITS), "resource_caps": dict(admission.CAPS),
        "algorithm": dict(admission.ALGORITHM), "reference_rectangles": [],
        "protected_roots": [],
    }


class MatrixOnlyContracts(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=os.environ.get("SAVED_PENCIL_TEST_ROOT"))

    def test_two_closed_subtypes_keep_legacy_source_and_roles(self):
        for coarse, interface in ((True, wrapper.MATRIX_ONLY_INTERFACE),
                                  (False, wrapper.SAVED_INTERFACE)):
            with self.subTest(coarse=coarse):
                packet = policy(coarse=coarse)
                profile, _ = admission.packet_policy(packet)
                self.assertEqual(profile["matrix_only"], coarse)
                self.assertEqual(len(profile["roles"]), 24 if coarse else 30)
                self.assertEqual(wrapper.saved_packet_profile(interface, packet), profile["source"])

    def test_wrong_anchor_source_counts_and_policy_refuse_before_input_io(self):
        faults = (
            ("schema", lambda p: p.update(schema="unbounded-saved-spectrum")),
            ("source", lambda p: p["authority"].update(source_commit=admission.SOURCE)),
            ("anchor", lambda p: p["model"]["physics"].update(ca_in=-0.00315)),
            ("nnz", lambda p: p["target"].update(j_nnz=10488602)),
            ("sector", lambda p: p["target"].update(m=1)),
            ("gate", lambda p: p["gates"].update(backward=1e-5)),
            ("cap", lambda p: p["resource_caps"].update(memory_bytes=32 * 2**30)),
            ("kernel", lambda p: p["source_closure"]["numerics.py"].update(sha256="0" * 64)),
            ("rectangle", lambda p: p.update(reference_rectangles=[{"fitted": True}])),
        )
        for name, change in faults:
            with self.subTest(fault=name):
                packet = policy()
                change(packet)
                with self.assertRaises(admission.AdmissionFailure):
                    admission.packet_policy(packet)

    def test_geometry_borrow_and_mixed_wrapper_subtypes_are_refused(self):
        for role in sorted(admission.GEOMETRY_ROLES):
            with self.subTest(role=role):
                packet = policy()
                packet["inputs"][role] = {"path": "borrowed-old-case"}
                with self.assertRaisesRegex(admission.AdmissionFailure, "bounded input roles"):
                    admission.packet_policy(packet)
                with self.assertRaisesRegex(ValueError, "input roles"):
                    wrapper.saved_packet_profile(wrapper.MATRIX_ONLY_INTERFACE, packet)
        with self.assertRaises(ValueError):
            wrapper.saved_packet_profile(wrapper.SAVED_INTERFACE, policy())
        with self.assertRaises(ValueError):
            wrapper.saved_packet_profile(wrapper.MATRIX_ONLY_INTERFACE, policy(coarse=False))

    def test_matrix_reference_is_empty_and_has_no_fitted_band(self):
        reference = {"schema": "saved-paper-matrix-only-reference-v1", "ca_in": -0.0038,
                     "reference_rectangles": [], "purpose": "pencil-level-challenger-search"}
        admission.matrix_only_reference(reference, [])
        for key, value in (("ca_in", -0.00315), ("reference_rectangles", [{"band": []}])):
            changed = copy.deepcopy(reference)
            changed[key] = value
            with self.assertRaises(admission.AdmissionFailure):
                admission.matrix_only_reference(changed, [])

    def test_matrix_only_lifts_never_call_geometry_diagnosis(self):
        def forbidden(*args):
            raise AssertionError("geometry diagnosis called without descriptors")
        prepared = {"geometry": None}
        record = search.saved_geometry_diagnostics(
            SimpleNamespace(geometry_diagnostics=forbidden), prepared, {}, object())
        self.assertEqual(record["status"], "explicitly_unavailable_matrix_only")
        self.assertIs(record["native_physical_mode_qualification"], False)

    def test_legacy_geometry_delegation_is_preserved(self):
        seen = []
        vector, matrix, geometry, helper = object(), object(), {}, {}
        prepared = {"geometry": geometry, "j": matrix,
                    "layout": {"globals": {"gas_volume/pressure": [316988]}}}
        def diagnose(*args):
            seen.append(args)
            return {"legacy": True}
        result = search.saved_geometry_diagnostics(SimpleNamespace(geometry_diagnostics=diagnose),
                                                  prepared, helper, vector)
        self.assertEqual(result, {"legacy": True})
        self.assertEqual(seen, [(vector, geometry, helper, matrix, 316988)])

    def child_record(self, parent, *, escalate=None):
        record = {"post_input_hashes_verified": True, "no_native_import": True,
                  "second_shift_automatic": False, "absence_claim_allowed": False,
                  "exact_root_exclusion_certified": False, "finite_spectrum_complete": False,
                  "global_dominance_certified": False, "physical_mode_qualification": False}
        if escalate:
            record[escalate] = True
        store = search.Store(parent / "result")
        store.event("prepared", packet_sha256="c" * 64)
        store.finish("no_additional_challenger_found_among_numerically_classified_checked_roots", record)
        return {"output": parent, "packet_record": {"sha256": "c" * 64}}

    def test_truthful_completion_stays_uncertified(self):
        with self.temporary() as temporary:
            prepared = self.child_record(Path(temporary))
            result, _ = wrapper.child_completion(prepared)
            for key in ("absence_claim_allowed", "exact_root_exclusion_certified",
                        "finite_spectrum_complete", "global_dominance_certified", "physical_mode_qualification"):
                self.assertIs(result[key], False)
        for key in ("absence_claim_allowed", "finite_spectrum_complete",
                    "global_dominance_certified", "physical_mode_qualification"):
            with self.subTest(escalation=key), self.temporary() as temporary:
                prepared = self.child_record(Path(temporary), escalate=key)
                with self.assertRaisesRegex(ValueError, "unestablished certificate"):
                    wrapper.child_completion(prepared)

    def test_regular_hash_and_native_denial_guards(self):
        with self.temporary() as temporary:
            file = Path(temporary) / "tiny.json"
            file.write_text("{}")
            with self.assertRaises(admission.AdmissionFailure):
                admission.read_regular(file, "0" * 64)
            alias = Path(temporary) / "alias.json"
            alias.symlink_to(file)
            with self.assertRaises(admission.AdmissionFailure):
                admission.read_regular(alias)
        guard = search.NativeDeny()
        with self.assertRaises(ImportError):
            guard.find_spec("pyoomph", None)
        search.no_native()


if __name__ == "__main__":
    unittest.main()
