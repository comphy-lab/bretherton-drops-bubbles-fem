"""Synthetic stream/API contracts; native imports and scientific claims are denied."""

from __future__ import annotations

import copy
import importlib.abc
import math
import os
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph.") or "_pyoomph" in fullname:
            raise AssertionError("native import denied in synthetic census suite")


guard = NoNative()
sys.meta_path.insert(0, guard)
try:
    from test_paper_dynamic_jump import Remesher, Template

    from runners import paper_dynamic_capture as capture
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import paper_dynamic_state as state
finally:
    sys.meta_path.remove(guard)


@pytest.fixture(autouse=True)
def deny_native_imports():
    blocker = NoNative()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def integer(value):
    return struct.pack("<q", value)


def string(value):
    return integer(len(value)) + value.encode("ascii")


def scalar(value):
    return struct.pack("<d", value)


def npy(dtype, shape, data):
    header = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode()
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    codecs = {"<f8": "d", "<i8": "q", "<i4": "i"}
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"".join(struct.pack("<" + codecs[dtype], v) for v in data)


def full_stream(witness, mesh="mesh.msh", compression=6):
    def array(dtype, shape, data):
        if compression is None:
            return npy(dtype, (1, *shape), data)
        compressed = zlib.compress(npy(dtype, shape, data), compression)
        return integer(len(compressed)) + compressed

    def record_arrays(keys, lengths, data, width):
        return (array("<i8", (len(keys), width), [v for k in keys for v in k])
                + array("<i4", (len(lengths),), lengths) + array("<f8", (len(data),), data))

    body = string("pyoomph_dump") + string("0.1.6") + string("global")
    body += scalar(witness["time"]) + integer(1) + integer(0) + integer(-100 if compression is None else compression)
    body += integer(1) + string(mesh) + integer(1) + integer(0)
    body += integer(len(witness["dts"])) + b"".join(map(scalar, witness["dts"]))
    body += integer(witness["counter"]) + scalar(witness["dts"][0]) + scalar(math.nan)
    domains, records = witness["geometry"]["domains"], {r["name"]: r for r in witness["data"]}
    body += integer(4)
    for name in ("gas", "gas_volume", "globals", "liquid"):
        body += string(name)
        view = domains[name]
        if view["kind"] == "ode":
            fields = sorted((records[ref]["fields"][0].split("/")[-1], records[ref]["rows"])
                            for refs in view["internal_references"] for ref in refs)
            body += integer(len(fields))
            for field, rows in fields:
                body += string(field) + integer(len(rows)) + b"".join(scalar(row[0]) for row in rows)
        else:
            roots = sorted({key[0] for key in view["element_keys"]})
            body += array("<i8", (len(roots),), roots)
            body += array("<i4", (len(roots),), [1] * len(roots))
            body += array("<i4", (len(roots),), [0] * len(roots))
            body += record_arrays(view["node_keys"], view["nodal_lengths"], view["nodal_buffer"], 3)
            body += record_arrays(view["element_keys"], view["elemental_lengths"], view["elemental_buffer"], 2)
            body += integer(0)
    interfaces = sorted(name for name in domains if "/" in name)
    body += integer(len(interfaces))
    for name in interfaces:
        view = domains[name]
        body += string(name) + record_arrays(view["element_keys"], view["elemental_lengths"], view["elemental_buffer"], 3)
    body += integer(len(interfaces)) + b"".join(string(name) + integer(0) for name in interfaces)
    parameters = witness["policy"]["all_parameters"]
    body += integer(len(parameters)) + b"".join(string(name) + scalar(value) for name, value in sorted(parameters.items()))
    body += integer(1) + string(witness["bc_cache"]) + integer(0) + integer(0) + string("EOF_pyoomph")
    return body


class Data:
    def __init__(self, equations, *, pinned=None, base=0):
        self.equations = equations
        self.pins = [e < 0 for e in equations] if pinned is None else pinned
        self.rows = [[float(base + 10 * t + i) for i in range(len(equations))] for t in range(7)]

    def nvalue(self):
        return len(self.equations)

    def ntstorage(self):
        return len(self.rows)

    def eqn_number(self, i):
        return self.equations[i]

    def is_pinned(self, i):
        return self.pins[i]

    def value_at_t(self, t, i):
        return self.rows[t][i]


class Node(Data):
    def __init__(self, equations, positions, base):
        super().__init__(equations, base=base)
        self.position = Data(positions, base=base + 100)
        self.copy = self.hanging = False

    def variable_position_pt(self):
        return self.position

    def ndim(self):
        return 2

    def x(self, i):
        return self.position.rows[0][i]

    def x_lagr(self, i):
        return float(i)

    def is_a_copy(self):
        return self.copy

    def is_hanging(self):
        return self.hanging


class Element:
    def __init__(self, data=(), external=()):
        self.data, self.external = list(data), list(external)

    def ninternal_data(self):
        return len(self.data)

    def nexternal_data(self):
        return len(self.external)

    def internal_data_pt(self, i):
        return self.data[i]

    def external_data_pt(self, i):
        return self.external[i]

    def get_initial_cartesian_nondim_size(self):
        return 1.25

    def get_initial_quality_factor(self):
        return 0.75


class Mesh:
    def __init__(self, nodes, element, *, interface=False):
        self.nodes, self.element, self.interface = nodes, element, interface
        self._interfacemeshes, self._tracers = {}, {}
        self.root, self.path = 1, 1

    def nnode(self):
        return len(self.nodes)

    def nelement(self):
        return 1

    def node_pt(self, i):
        return self.nodes[i]

    def element_pt(self, i):
        assert i == 0
        return self.element

    def _get_dirichlet_active_flags(self):
        return [True]

    def get_field_information(self):
        return {"velocity": "C2", "pressure": "C1"}

    def get_nodal_field_indices(self):
        return {"velocity": 0, "pressure": 1}

    def get_element_node_indices(self):
        return list(range(len(self.nodes))), len(self.nodes)

    def get_element_structural_keys(self):
        return [self.root, self.path]

    def get_interface_element_structural_keys(self):
        return [self.root, self.path, 0]

    def save_nodal_state(self):
        blocks = []
        for node in self.nodes:
            block = [row[i] for i in range(2) for row in node.position.rows]
            block += [node.x_lagr(i) for i in range(2)]
            block += [row[i] for i in range(node.nvalue()) for row in node.rows]
            blocks.append(block)
        return [v for b in blocks for v in b], [len(b) for b in blocks]

    def save_elemental_state(self):
        block = [row[i] for data in self.element.data for i in range(data.nvalue()) for row in data.rows]
        block += [self.element.get_initial_cartesian_nondim_size(), self.element.get_initial_quality_factor()]
        return block, [len(block)]


class OdeMesh(Mesh):
    def __init__(self, name, data):
        super().__init__([], Element([data]))
        self.field = name

    def get_element(self):
        return self.element


def ode_numpy(self):
    return [], {self.field: 0}


class Stepper:
    def ntstorage(self):
        return 7

    def nprev_values(self):
        return 2

    def is_steady(self):
        return False

    def time_pt(self):
        return self

    def time(self, index=0):
        return 4.0 - index * 0.25

    def dt(self, i):
        return 0.25

    def ndt(self):
        return 2

    def get_num_unsteady_steps_done(self):
        return 4

    def weightBDF1(self, i, j):
        return float(i + j)

    weightBDF2 = weightBDF1
    weightNewmark2 = weightBDF1


class Problem:
    def __init__(self, mesh_path):
        self.timestepper, self._hooks = Stepper(), []
        self._dof_selector = self._dof_selector_used = None
        self.initial_adaption_steps, self.project_initial_conditions = 0, False
        self.initialised = True
        self.default_timestepping_scheme, self._last_bc_setting = "BDF2", "transient"
        self._output_step, self._continue_section_step = 1, 0
        self._suggested_next_dt, self._run_statement_endtime = 0.25, None
        self._bifurcation_reactivation_after_adaptation = None
        self._azimuthal_mode_param_m = SimpleNamespace(value=0.0)
        self.parameters = {"Ca_in": -0.004, "Bo": 1.56, "_azimuthal_mode_m": 0.0}
        self._first_step, self._taken_already_an_unsteady_step, self._last_step_was_stationary = False, True, False
        self.eigen_data_in_states, self.continuation_data_in_states = True, False
        eq = 0
        meshes = {}
        for name in ("gas", "liquid"):
            nodes = []
            for i in range(3):
                nodes.append(Node([eq, -1], [eq + 1, eq + 2], 100 * i))
                eq += 3
            meshes[name] = Mesh(nodes, Element())
        for name, field in (("gas_volume", "pressure"), ("globals", "U")):
            meshes[name] = OdeMesh(field, Data([eq], base=1000))
            meshes[name].element._ode_elem_to_numpy = lambda mesh=meshes[name]: ode_numpy(mesh)
            eq += 1
        self._ndof, self._meshdict = eq, meshes
        meshes["gas"]._interfacemeshes["interface"] = Mesh(meshes["gas"].nodes, Element(), interface=True)
        meshes["gas"].element.external = [meshes["globals"].element.data[0]]
        original, active = Template(self), Template(self)
        original._template_override = active
        original.remesher = active.remesher = Remesher(active)
        active._meshfile = str(mesh_path)
        self._meshtemplate_list, self._domains_remesh_on_inversion, self._domains_to_remesh = [original], {original, active}, set()
        for name in ("gas", "liquid"):
            meshes[name]._templatemesh = active
        self.adapter = capture.Capture(capture.NativeTypes(Problem, Stepper, OdeMesh, Template, Remesher))

    def ndof(self):
        return self._ndof

    def get_dof_description(self):
        return list(range(self._ndof)), ["synthetic/" + str(i) for i in range(self._ndof)]

    def ntime_stepper(self):
        return 1

    def time_stepper_pt(self, i):
        return self.timestepper

    def is_distributed(self):
        return False

    def is_initialised(self):
        return self.initialised

    def get_bifurcation_tracking_mode(self):
        return ""

    def get_custom_assembler(self):
        return None

    def _get_solved_residual(self):
        return ""

    def get_scaling(self, name):
        return 1.0

    def get_current_time(self, **kwargs):
        return self.timestepper.time()

    def get_global_parameter_names(self):
        return list(self.parameters)

    def get_global_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def get_last_eigenvalues(self):
        return []

    def save_state(self, path, quiet=True):
        mesh = self._meshtemplate_list[0]._get_template()._meshfile
        Path(path).write_bytes(full_stream(self.adapter(self), os.path.relpath(mesh, Path(path).parent)))


@pytest.fixture
def problem(tmp_path):
    mesh = tmp_path / "source.msh"
    mesh.write_bytes(b"synthetic mesh; never loaded")
    return Problem(mesh)


def test_complete_raw_census_proves_zero_spatial_internal_and_preserves_aliases_and_pins(problem):
    witness = problem.adapter(problem)
    assert witness["census"]["counts"] == {"nodal": 6, "position": 6, "internal": 0, "ode": 2}
    assert len(witness["census"]["occurrences"]) > len(witness["data"])
    assert witness["census"]["external"]
    assert witness["census"]["ownership"]["pointer_observed"] is False
    pinned = next(block for block in witness["data"] if block["kind"] == "nodal")
    assert pinned["pins"][1] and pinned["equations"][1] == -1
    assert [row[1] for row in pinned["rows"]] == [1.0 + 10 * i for i in range(7)]
    assert runtime.validate_witness(witness)["census"]["counts"]["internal"] == 0


def test_fresh_initialisation_policy_prevents_the_native_default_refinement_trap(problem):
    problem.initialised, problem.initial_adaption_steps = False, None
    receipt = capture.prepare_initialisation(problem, problem.adapter.types)
    assert problem.initial_adaption_steps == 0 and not problem.initialised
    assert receipt["native_initialisation_called"] is False
    problem.project_initial_conditions = True
    with pytest.raises(ValueError, match="projected initial conditions"):
        capture.prepare_initialisation(problem, problem.adapter.types)
    with pytest.raises(ValueError, match="refinement/projection"):
        problem.adapter(problem)


@pytest.mark.parametrize("compression", [None, 0, 6, 9])
def test_full_independent_numeric_stream_and_portable_rewrite(problem, compression):
    witness = problem.adapter(problem)
    raw = full_stream(witness, compression=compression)
    decoded = state.decode(raw)
    assert capture.cross_check(decoded, witness)["complete_numeric_cross_check_passed"] is True
    portable = state.portable_bytes(raw, decoded, "closed.msh")
    start, end = decoded["templates"][0]["string_range"]
    next_start, next_end = state.decode(portable)["templates"][0]["string_range"]
    assert raw[:start] == portable[:next_start] and raw[end:] == portable[next_end:]
    assert capture.cross_check(state.decode(portable), witness)["native_load_qualified"] is False


@pytest.mark.parametrize("fault", ["clock", "dt", "pin_history", "ode_history", "element_reference"])
def test_independent_cross_check_rejects_complete_history_and_control_drift(problem, fault):
    witness = problem.adapter(problem)
    decoded = state.decode(full_stream(witness))
    changed = copy.deepcopy(witness)
    if fault == "clock":
        changed["time"] += 0.25
    elif fault == "dt":
        changed["dts"][1] += 0.25
    elif fault == "pin_history":
        next(block for block in changed["data"] if block["kind"] == "nodal")["rows"][4][1] += 1
    elif fault == "ode_history":
        next(block for block in changed["data"] if block["kind"] == "ode")["rows"][5][0] += 1
    else:
        changed["geometry"]["domains"]["gas"]["element_reference"][0][1] += 1
    with pytest.raises(ValueError):
        capture.cross_check(decoded, changed)


@pytest.mark.parametrize("fault", ["extra_stepper", "custom_hook", "copy", "hanging", "refined", "unowned_external"])
def test_source_ownership_rejects_uncovered_runtime_structures(problem, fault):
    if fault == "extra_stepper":
        problem.ntime_stepper = lambda: 2
    elif fault == "custom_hook":
        problem._hooks = [object()]
    elif fault in {"copy", "hanging"}:
        setattr(problem._meshdict["gas"].nodes[0], fault, True)
    elif fault == "refined":
        problem._meshdict["gas"].path = 2
    else:
        problem._meshdict["gas"].element.external.append(Data([-1]))
    with pytest.raises(ValueError):
        problem.adapter(problem)


def test_census_rejects_missing_pinned_owner_even_when_free_equations_are_covered(problem):
    witness = problem.adapter(problem)
    witness["census"]["occurrences"] = []
    with pytest.raises(ValueError, match="traversal coverage"):
        runtime.validate_witness(witness)


@pytest.mark.parametrize("fault", ["truncated", "trailing", "oversized", "pickle", "compressed_bomb"])
def test_numeric_decoder_refuses_unsafe_or_incomplete_formats(problem, fault):
    raw = full_stream(problem.adapter(problem))
    if fault == "truncated":
        raw = raw[:-8]
    elif fault == "trailing":
        raw += b"unexpected"
    elif fault == "oversized":
        with pytest.raises(ValueError, match="byte bound"):
            state.decode(raw, limits=state.Limits(max_bytes=8))
        return
    elif fault == "pickle":
        header = repr({"descr": "|O", "fortran_order": False, "shape": (1,)}).encode() + b"\n"
        stream = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"never unpickle this"
        reader = state.Reader(stream, state.Limits())
        with pytest.raises(ValueError, match="object"):
            reader.npy(reader.stream)
        return
    else:
        compressed = zlib.compress(b"x" * 1024)
        reader = state.Reader(integer(len(compressed)) + compressed, state.Limits(max_bytes=128))
        reader.compression = 6
        with pytest.raises(ValueError, match="compressed array"):
            reader.array()
        return
    with pytest.raises(ValueError):
        state.decode(raw)


def test_numeric_decoder_keeps_fortran_order_and_rejects_duplicate_header_keys():
    header = repr({"descr": "<f8", "fortran_order": True, "shape": (2, 3)}).encode() + b"\n"
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    raw += b"".join(scalar(v) for v in (0, 3, 1, 4, 2, 5))
    reader = state.Reader(raw, state.Limits())
    assert reader.npy(reader.stream).values == tuple(float(v) for v in range(6))
    header = b"{'descr':'<f8','descr':'|O','fortran_order':False,'shape':(1,)}\n"
    reader = state.Reader(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header, state.Limits())
    with pytest.raises(ValueError, match="duplicate"):
        reader.npy(reader.stream)


def test_full_decoder_consumes_optional_eigen_and_continuation_without_qualifying_them(problem):
    witness = problem.adapter(problem)
    raw = full_stream(witness)
    start = state.decode(raw)["eigen_section_start"]

    def array(dtype, shape, items):
        compressed = zlib.compress(npy(dtype, shape, items), 6)
        return integer(len(compressed)) + compressed

    tail = integer(1) + string("transient") + integer(1)
    tail += array("<f8", (1,), [0.0]) + array("<f8", (1, witness["ndof"]), [0.0] * witness["ndof"])
    tail += integer(1) + array("<i8", (1,), [0])
    tail += integer(1) + array("<f8", (witness["ndof"],), [0.0] * witness["ndof"])
    tail += scalar(1.0) + scalar(1.0) + string("EOF_pyoomph")
    decoded = state.decode(raw[:start] + tail)
    assert decoded["eigen"]["count"] == 1 and decoded["continuation"]["direction"].shape == (witness["ndof"],)
    with pytest.raises(ValueError, match="zero-eigen"):
        capture.cross_check(decoded, witness)


def test_portable_retention_keeps_all_mesh_history_policy_and_cross_check(problem, tmp_path):
    witness = problem.adapter(problem)
    output = tmp_path / "retained"
    output.mkdir()
    retain = capture.Retain(problem.adapter, output, lambda p: {"synthetic": True, "native": False})
    result = retain(problem, {"accepted_step_index": 4}, witness)
    decoded = state.decode(Path(result["state"]["path"]).read_bytes())
    assert decoded["templates"][0]["stored_path"].startswith("mesh-")
    assert (Path(result["state"]["path"]).parent / decoded["templates"][0]["stored_path"]).is_file()
    assert problem.adapter(problem) == witness
    assert Path(result["numeric_cross_check"]["path"]).is_file()
    with pytest.raises(FileExistsError):
        retain(problem, {"accepted_step_index": 4}, witness)


def test_retention_refuses_a_profile_that_materialises_inactive_pressure(problem, tmp_path):
    witness = problem.adapter(problem)
    output = tmp_path / "retained"
    output.mkdir()

    def bad_profile(p):
        p._meshdict["gas"].nodes[0].rows[0][1] += 1
        return {"synthetic": True}

    with pytest.raises(ValueError, match="changed physical histories"):
        capture.Retain(problem.adapter, output, bad_profile)(problem, {"accepted_step_index": 4}, witness)
