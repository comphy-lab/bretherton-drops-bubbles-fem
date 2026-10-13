"""Raw census and portable retention for the closed, pinned TH construction.

Native objects/types must be supplied by an independently admitted caller.
This module never imports pyoomph and cannot authorise native execution.
The source-derived stepper mapping is not a native Data-pointer observation.
"""

from __future__ import annotations

import copy
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from runners import paper_dynamic_jump as study
from runners import paper_dynamic_jump_runtime as runtime
from runners import paper_dynamic_state as state

require = study.require
KINDS = ("nodal", "position", "internal", "ode")


@dataclass(frozen=True)
class NativeTypes:
    problem: type
    stepper: type
    ode_mesh: type
    template: type
    remesher: type


def prepare_initialisation(problem, types):
    """Freeze the registered zero-refinement policy before any native initialise.

    Pinned initialise otherwise replaces initial_adaption_steps=None with
    max_refinement_level. This helper never initialises, loads or solves.
    """
    require(type(problem) is types.problem and not problem.is_initialised(),
            "prepare the selected fresh problem before native initialisation")
    require(problem.project_initial_conditions is False, "projected initial conditions are unsupported")
    previous = problem.initial_adaption_steps
    problem.initial_adaption_steps = 0
    return {"schema": "stone-capture-initialisation-policy-v1", "previous_initial_adaption_steps": previous,
            "initial_adaption_steps": 0, "project_initial_conditions": False,
            "native_initialisation_called": False}


def values(array):
    return array.tolist() if hasattr(array, "tolist") else list(array)


def chunks(items, width):
    require(type(width) is int and width > 0 and len(items) % width == 0, "invalid native buffer stride")
    return [items[i:i + width] for i in range(0, len(items), width)]


def block_map(keys, lengths, data):
    require(len(keys) == len(lengths) and len({tuple(k) for k in keys}) == len(keys)
            and all(type(n) is int and n >= 0 for n in lengths) and sum(lengths) == len(data),
            "native structural blocks are incomplete")
    result, cursor = {}, 0
    for key, length in zip(keys, lengths, strict=True):
        result[tuple(key)] = data[cursor:cursor + length]
        cursor += length
    return result


def node_keys(element_keys, connectivity, count):
    result = [None] * count
    for key, nodes in zip(element_keys, connectivity, strict=True):
        require(len(key) == 2 and key[0] >= 0 and key[1] == 1,
                "unassigned/refined structure is outside the primary construction contract")
        for local, index in enumerate(nodes):
            require(type(index) is int and -1 <= index < count, "invalid native connectivity index")
            if index >= 0:
                candidate = (*key, local)
                result[index] = candidate if result[index] is None else min(result[index], candidate)
    require(all(key is not None for key in result), "native node has no structural owner")
    return [list(key) for key in result]


def stepper_record(stepper):
    clock = stepper.time_pt()
    nt = int(stepper.ntstorage())
    return {"type": type(stepper).__module__ + "." + type(stepper).__qualname__,
            "order": 2, "order_origin": "a186_MultiTimeStepper_order_definition",
            "ntstorage": nt, "nprev_values": int(stepper.nprev_values()),
            "is_steady": bool(stepper.is_steady()), "time": float(clock.time()),
            "time_history": [float(clock.time(i)) for i in range(clock.ndt() + 1)],
            "dts": [float(clock.dt(i)) for i in range(clock.ndt())],
            "counter": int(stepper.get_num_unsteady_steps_done()),
            "weights": {name: [[float(getattr(stepper, "weight" + name)(order, slot))
                                for slot in range(nt)] for order in range(3)]
                        for name in ("BDF1", "BDF2", "Newmark2")}}


def validate_census(census, data, steppers):
    require(isinstance(census, dict) and set(census) == {"schema", "counts", "occurrences", "external",
            "ownership", "enumerated_meshes"} and census["schema"] == "stone-raw-data-census-v1",
            "exhaustive native census required")
    require(set(census["counts"]) == set(KINDS), "every census category must be recorded")
    actual = {kind: sum(block["kind"] == kind for block in data) for kind in KINDS}
    require(census["counts"] == actual and all(type(n) is int and n >= 0 for n in actual.values()),
            "census category counts differ")
    names = {block["name"] for block in data}
    occurrences = census["occurrences"]
    require(isinstance(occurrences, list) and occurrences and all(isinstance(row, dict)
            and set(row) == {"site", "kind", "owner"} and row["owner"] in names and row["kind"] in KINDS
            and isinstance(row["site"], str) and row["site"] for row in occurrences)
            and len({row["site"] for row in occurrences}) == len(occurrences)
            and {row["owner"] for row in occurrences} == names, "native traversal coverage is incomplete")
    blocks = {block["name"]: block for block in data}
    require(all(blocks[row["owner"]]["kind"] == row["kind"] for row in occurrences),
            "alias changed Data category")
    require(isinstance(census["enumerated_meshes"], list) and census["enumerated_meshes"]
            and len(set(census["enumerated_meshes"])) == len(census["enumerated_meshes"]),
            "complete mesh traversal inventory required")
    require(isinstance(census["external"], list) and all(isinstance(row, dict)
            and set(row) == {"site", "owner"} and row["owner"] in names for row in census["external"]),
            "unresolved external Data owner")
    ownership = census["ownership"]
    require(isinstance(ownership, dict) and set(ownership) == {"origin", "pyoomph_pin", "pointer_observed",
            "registered_count"} and ownership["origin"] in {"exact_pinned_primary_factory_chain", "synthetic"}
            and ownership["pyoomph_pin"] == study.PYOOMPH_PIN and ownership["pointer_observed"] is False
            and ownership["registered_count"] == len(steppers), "stepper ownership proof is unbound")
    if ownership["origin"] == "exact_pinned_primary_factory_chain":
        require(len(steppers) == 1 and all(block.get("stepper_index") == 0 for block in data),
                "source-derived mapping does not cover this registry")
        primary = steppers[0]
        require(primary.get("order_origin") == "a186_MultiTimeStepper_order_definition"
                and primary.get("nprev_values") == 2 and isinstance(primary.get("time_history"), list)
                and len(primary["time_history"]) == len(primary.get("dts", [])) + 1
                and primary.get("time") == primary["time_history"][0]
                and isinstance(primary.get("weights"), dict)
                and set(primary["weights"]) == {"BDF1", "BDF2", "Newmark2"},
                "complete pinned primary clock/weight banks are required")
        for bank in primary["weights"].values():
            require(isinstance(bank, list) and len(bank) == 3
                    and all(isinstance(row, list) and len(row) == primary["ntstorage"] for row in bank),
                    "complete stepper weight bank shape required")
            for row in bank:
                for value in row:
                    study.number(value, "native stepper weight", signed=True)
    for block in data:
        index = block.get("stepper_index")
        require(type(index) is int and 0 <= index < len(steppers)
                and len(block["rows"]) == steppers[index]["ntstorage"], "Data history/stepper binding differs")


class Capture:
    """Source implementation; native qualification is a separate admission gate."""

    def __init__(self, types):
        self.types = types

    @property
    def template_type(self):
        return self.types.template

    @property
    def remesher_type(self):
        return self.types.remesher

    def construction(self, problem):
        require(type(problem) is self.types.problem and type(problem.timestepper) is self.types.stepper,
                "foreign problem/stepper construction")
        require(problem.ntime_stepper() == 1 and problem.time_stepper_pt(0) is problem.timestepper,
                "primary-only construction does not cover the actual registry")
        require(not problem.is_distributed() and not problem._hooks,
                "distributed/custom hook construction is unsupported")
        require(problem._dof_selector is None and problem._dof_selector_used is None,
                "custom or pending DOF selection is unsupported")
        require(type(problem.initial_adaption_steps) is int and problem.initial_adaption_steps == 0
                and problem.project_initial_conditions is False,
                "initial refinement/projection policy was not frozen before initialisation")
        require(set(problem._meshdict) == {"gas", "liquid", "gas_volume", "globals"},
                "selected finite-gas mesh inventory changed")
        require(problem.timestepper.nprev_values() == 2 and problem.timestepper.ntstorage() in (5, 7),
                "primary storage does not match the pinned factory")
        require(problem.get_bifurcation_tracking_mode() == "" and problem.get_custom_assembler() is None
                and problem._bifurcation_reactivation_after_adaptation is None and problem._get_solved_residual() == ""
                and float(problem._azimuthal_mode_param_m.value) == 0.0,
                "only the unaugmented base construction is covered")
        require(float(problem.get_scaling("temporal")) == 1.0, "temporal scaling changed")
        require(problem.timestepper.time_pt().time() == problem.get_current_time(dimensional=False, as_float=True),
                "registered primary clock is not the problem clock")
        runtime.owned_quality_templates(problem, self)

    def __call__(self, problem):
        self.construction(problem)
        records, objects, occurrences, external, geometry = [], [], [], [], {}
        identity_index = {}  # In-process wrapper lookup only; never persisted as a native pointer.
        primary = stepper_record(problem.timestepper)

        def add(obj, site, kind, field_names=None):
            index = identity_index.get(id(obj))
            if index is None:
                nv, nt = int(obj.nvalue()), int(obj.ntstorage())
                require(nv > 0 and nt == primary["ntstorage"], "uncovered empty/foreign Data history")
                name = site
                rows = [[float(obj.value_at_t(t, i)) for i in range(nv)] for t in range(nt)]
                records.append({"name": name, "kind": kind,
                    "fields": field_names if field_names is not None else [kind + "/raw_column_" + str(i) for i in range(nv)],
                    "rows": rows, "pins": [bool(obj.is_pinned(i)) for i in range(nv)],
                    "equations": [int(obj.eqn_number(i)) for i in range(nv)], "stepper_index": 0})
                objects.append(obj)
                identity_index[id(obj)] = len(objects) - 1
            else:
                name = records[index]["name"]
                require(records[index]["kind"] == kind, "Data reused across incompatible categories")
            occurrences.append({"site": site, "kind": kind, "owner": name})
            return name

        def walk(name, mesh, *, ode=False):
            require(not getattr(mesh, "_tracers", {}), "tracers are outside the selected construction")
            view = {"kind": "ode" if ode else "spatial", "nodes": [], "lagrangian": [],
                    "node_references": [], "internal_references": [], "element_reference": [],
                    "bc_active": values(mesh._get_dirichlet_active_flags())}
            if ode:
                require(type(mesh) is self.types.ode_mesh and mesh.nelement() == 1, "foreign ODE mesh")
                _, indices = mesh.get_element()._ode_elem_to_numpy()
                require(sorted(indices.values()) == list(range(mesh.get_element().ninternal_data())),
                        "ODE field map does not cover its actual Data")
                ode_names = {index: name + "/" + field for field, index in indices.items()}
            else:
                view["field_spaces"] = dict(mesh.get_field_information())
                def space_name(space):
                    while space.startswith("../"):
                        space = space.removeprefix("../")
                    return space
                require(all(space_name(space) in {"C1", "C2", "Pos"} for space in view["field_spaces"].values()),
                        "uncovered discontinuous or bubble-enriched field space")
                view["code_field_indices"] = dict(mesh.get_nodal_field_indices())
                connection, stride = mesh.get_element_node_indices()
                view.update(connectivity=values(connection), stride=int(stride))
                nb, nl = map(values, mesh.save_nodal_state())
                eb, el = map(values, mesh.save_elemental_state())
                view.update(nodal_buffer=nb, nodal_lengths=nl, elemental_buffer=eb, elemental_lengths=el)
                if "/" in name:
                    view["element_keys"] = chunks(values(mesh.get_interface_element_structural_keys()), 3)
                    keys = [[i] for i in range(mesh.nnode())]  # References point to canonical bulk Data.
                else:
                    view["element_keys"] = chunks(values(mesh.get_element_structural_keys()), 2)
                    keys = node_keys(view["element_keys"], chunks(view["connectivity"], int(stride)), mesh.nnode())
                    view["node_keys"] = keys
                require(len(nl) == mesh.nnode() and len(el) == mesh.nelement(), "native serializer census differs")
                for index in range(mesh.nnode()):
                    node = mesh.node_pt(index)
                    require(not node.is_a_copy() and not node.is_hanging(), "periodic/hanging node construction is unsupported")
                    key = ":".join(map(str, keys[index]))
                    value_ref = add(node, name + "/node/" + key + "/values", "nodal")
                    position = node.variable_position_pt()
                    require(position.nvalue() == node.ndim() == 2, "higher-order/non-axisymmetric position storage")
                    position_ref = add(position, name + "/node/" + key + "/positions", "position", ["mesh_x", "mesh_y"])
                    nlag = nl[index] - node.ntstorage() * (node.ndim() + node.nvalue())
                    require(nlag == 2, "Lagrangian serialization layout differs")
                    view["nodes"].append([float(node.x(i)) for i in range(2)])
                    view["lagrangian"].append([float(node.x_lagr(i)) for i in range(nlag)])
                    view["node_references"].append({"values": value_ref, "positions": position_ref})
            for index in range(mesh.nelement()):
                element = mesh.element_pt(index)
                refs = []
                for slot in range(element.ninternal_data()):
                    key = str(index) if ode else ":".join(map(str, view["element_keys"][index]))
                    site = name + "/element/" + key + "/internal/" + str(slot)
                    ref = add(element.internal_data_pt(slot), site, "ode" if ode else "internal",
                              [ode_names[slot]] if ode else None)
                    refs.append(ref)
                view["internal_references"].append(refs)
                if not ode:
                    view["element_reference"].append([float(element.get_initial_cartesian_nondim_size()),
                                                      float(element.get_initial_quality_factor())])
                for slot in range(element.nexternal_data()):
                    external.append((name + "/element/" + str(index) + "/external/" + str(slot), element.external_data_pt(slot)))
            geometry[name] = view
            if not ode:
                for child, interface in sorted(mesh._interfacemeshes.items()):
                    walk(name + "/" + child, interface)

        # Own all bulk/ODE objects before resolving interface and external aliases.
        for name, mesh in sorted(problem._meshdict.items()):
            walk(name, mesh, ode=name in {"gas_volume", "globals"})
        external_rows = []
        for site, obj in external:
            index = identity_index.get(id(obj))
            require(index is not None, "external Data has no proved owner")
            external_rows.append({"site": site, "owner": records[index]["name"]})
        types, names = problem.get_dof_description()
        nodes, lagrangian, connectivity = [], [], []
        for name in ("gas", "liquid"):
            view = geometry[name]
            offset = len(nodes)
            nodes.extend(view["nodes"])
            lagrangian.extend(view["lagrangian"])
            connectivity.extend(index + offset for index in view["connectivity"] if index >= 0)
        geometry = {"nodes": nodes, "lagrangian": lagrangian, "connectivity": connectivity, "domains": geometry}
        census = {"schema": "stone-raw-data-census-v1", "counts": {kind: sum(r["kind"] == kind for r in records) for kind in KINDS},
                  "occurrences": occurrences, "external": external_rows, "enumerated_meshes": sorted(geometry["domains"]),
                  "ownership": {"origin": "exact_pinned_primary_factory_chain", "pyoomph_pin": study.PYOOMPH_PIN,
                                "pointer_observed": False, "registered_count": 1}}
        witness = {"schema": "stone-jump-complete-physical-v3", "ndof": int(problem.ndof()), "data": records,
                   "geometry": geometry, "census": census,
                   "layout": {"ndof": int(problem.ndof()), "types": values(types), "dof_names": list(names)},
                   "parameters": {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")},
                   "time": primary["time"], "dts": primary["dts"], "counter": primary["counter"], "steppers": [primary],
                   "flags": {name: getattr(problem, name) for name in study.FLAGS}, "bc_cache": problem._last_bc_setting,
                   "mode": copy.deepcopy(runtime.BASE_MODE), "pending_quality_remesh": runtime.pending_remesh_policy(problem, self),
                   "policy": {"all_parameters": {name: float(problem.get_global_parameter(name).value)
                              for name in sorted(problem.get_global_parameter_names())},
                              "default_scheme": problem.default_timestepping_scheme,
                              "serialization": {"output_step": problem._output_step,
                                  "continue_step": problem._continue_section_step,
                                  "suggested_next_dt": problem._suggested_next_dt,
                                  "run_endtime": problem._run_statement_endtime,
                                  "remesher_counter": problem._meshtemplate_list[0]._get_template().remesher._cnt}}}
        require(witness["policy"]["default_scheme"] == "BDF2", "default scheme changed")
        return runtime.validate_witness(witness)


def packed(values_):
    return b"".join(struct.pack("<d", value) for value in values_)


def cross_check(decoded, witness):
    """Compare complete numeric payloads independently of the native reader."""
    require(decoded["time"] == witness["time"] and decoded["dts"] == witness["dts"]
            and decoded["counter"] == witness["counter"]
            and decoded["parameters"] == witness["policy"]["all_parameters"], "stream/live control clock differs")
    serialized = witness["policy"]["serialization"]
    require(all(decoded[key] == serialized[key] for key in ("output_step", "continue_step", "suggested_next_dt", "run_endtime"))
            and decoded["templates"][0]["remesher_counter"] == serialized["remesher_counter"],
            "stream/live serialization control metadata differs")
    require(not decoded["continuation"] and decoded["eigen"] == {"bc_cache": witness["bc_cache"], "count": 0},
            "retained state requires the zero-eigen/continuation BC policy")
    domains = witness["geometry"]["domains"]
    require(set(decoded["interfaces"]) == {name for name in domains if "/" in name}, "interface census differs")
    data = {row["name"]: row for row in witness["data"]}
    for name, view in domains.items():
        if view["kind"] == "ode":
            fields = {data[ref]["fields"][0].removeprefix(name + "/"): [r[0] for r in data[ref]["rows"]]
                      for refs in view["internal_references"] for ref in refs}
            require(decoded["meshes"][name]["fields"] == fields, "stream/live ODE history differs")
            continue
        # Compare the independent raw Data getter with the native serializer's
        # projection, including pinned/derived C1 columns and every history slot.
        projected_nodes = []
        for refs, lagrangian in zip(view["node_references"], view["lagrangian"], strict=True):
            position, nodal = data[refs["positions"]], data[refs["values"]]
            block = [row[column] for column in range(len(position["fields"])) for row in position["rows"]]
            block += lagrangian
            block += [row[column] for column in range(len(nodal["fields"])) for row in nodal["rows"]]
            projected_nodes.append(block)
        require([len(block) for block in projected_nodes] == view["nodal_lengths"]
                and packed([v for block in projected_nodes for v in block]) == packed(view["nodal_buffer"]),
                "raw Data histories differ from native nodal serialization")
        projected_elements = []
        for refs, reference in zip(view["internal_references"], view["element_reference"], strict=True):
            block = [row[column] for ref in refs for column in range(len(data[ref]["fields"])) for row in data[ref]["rows"]]
            projected_elements.append(block + reference)
        require([len(block) for block in projected_elements] == view["elemental_lengths"]
                and packed([v for block in projected_elements for v in block]) == packed(view["elemental_buffer"]),
                "raw internal/reference storage differs from native elemental serialization")
        if "/" in name:
            raw = decoded["interfaces"][name]
            keys = chunks(list(raw["keys"].values), 3)
            lengths, payload = raw["lengths"].values, raw["data"].values
        else:
            arrays = decoded["meshes"][name]["arrays"]
            require(arrays["roots"].values == tuple(sorted({key[0] for key in view["element_keys"]}))
                    and arrays["sig_lens"].values == (1,) * len(arrays["roots"].values)
                    and arrays["sig_data"].values == (0,) * len(arrays["roots"].values),
                    "stream/live refinement construction differs")
            raw_nodes = block_map(chunks(list(arrays["node_keys"].values), 3), list(arrays["node_lens"].values), list(arrays["node_data"].values))
            live_nodes = block_map(view["node_keys"], view["nodal_lengths"], view["nodal_buffer"])
            require(set(raw_nodes) == set(live_nodes) and all(packed(raw_nodes[k]) == packed(live_nodes[k]) for k in raw_nodes),
                    "stream/live complete nodal storage differs")
            keys = chunks(list(arrays["elem_keys"].values), 2)
            lengths, payload = arrays["elem_lens"].values, arrays["elem_data"].values
        raw_elements = block_map(keys, list(lengths), list(payload))
        live_elements = block_map(view["element_keys"], view["elemental_lengths"], view["elemental_buffer"])
        require(set(raw_elements) == set(live_elements)
                and all(packed(raw_elements[k]) == packed(live_elements[k]) for k in raw_elements),
                "stream/live complete elemental reference/history storage differs")
    return {"schema": "stone-native-numeric-cross-check-v1", "source_pin": study.PYOOMPH_PIN,
            "state_sha256": decoded["sha256"], "native_load_qualified": False,
            "complete_numeric_cross_check_passed": True}


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "retention authority must be an absolute regular file")
    return path.resolve(strict=True)


def write_new(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


class Retain:
    """Write one fresh portable row; no native dispatch or qualification is implied."""

    def __init__(self, capture, output, profile):
        self.capture, self.output, self.profile = capture, Path(output), profile
        require(self.output.is_absolute() and self.output.is_dir()
                and not any(p.is_symlink() for p in (self.output, *self.output.parents)), "fresh regular retention directory required")

    def __call__(self, problem, row, expected):
        require(self.capture(problem) == expected, "retention began from a different physical state")
        require(problem.eigen_data_in_states is True and problem.continuation_data_in_states is False,
                "register zero-eigen/continuation BC serialization before retention")
        require(len(problem.get_last_eigenvalues()) == 0, "retention must not write stale eigenpairs")
        directory = self.output / ("native-step-" + str(row["accepted_step_index"]))
        directory.mkdir(mode=0o700)
        temporary = directory / "unportable.dump"
        problem.save_state(str(temporary), quiet=True)
        original = regular(temporary).read_bytes()
        decoded = state.decode(original)
        cross_check(decoded, expected)
        _, owner = runtime.owned_quality_templates(problem, self.capture)
        mesh = regular(Path(owner._meshfile).absolute())
        reference = (directory / decoded["templates"][0]["stored_path"]).resolve(strict=True)
        require(reference == mesh and not decoded["templates"][0]["stored_path"].startswith("/"),
                "state mesh is not the live proved template authority")
        mesh_bytes = mesh.read_bytes()
        mesh_name = "mesh-" + hashlib.sha256(mesh_bytes).hexdigest() + ".msh"
        mesh_ref = write_new(directory / mesh_name, mesh_bytes)
        require(mesh.read_bytes() == mesh_bytes, "live mesh changed during retention")
        portable = state.portable_bytes(original, decoded, mesh_name)
        state_ref = write_new(directory / "state.dump", portable)
        decoded = state.decode(portable)
        receipt = cross_check(decoded, expected)
        inventory = {"state": state_ref, "mesh_reference_base": str(directory),
                     "meshes": [{"stored_path": mesh_name, **mesh_ref}]}
        physical = {"state": state_ref, "witness": expected}
        profile = self.profile(problem)
        require(self.capture(problem) == expected, "capture/save/profile changed physical histories or policy")
        encode = lambda value: (study.json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        result = {**row, "state": state_ref,
                  "mesh_inventory": write_new(directory / "mesh-inventory.json", encode(inventory)),
                  "physical_witness": write_new(directory / "physical-witness.json", encode(physical)),
                  "profile": write_new(directory / "profile.json", encode(profile)),
                  "numeric_cross_check": write_new(directory / "numeric-cross-check.json", encode(receipt))}
        temporary.unlink()  # This writer's temporary copy is now preserved portably and cross-checked.
        return result
