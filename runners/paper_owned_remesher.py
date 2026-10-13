"""Select one owned two-phase remesher without changing native registration."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import struct
from pathlib import Path

from runners import run_paper_volume_seed_remesh as seed

require, sha = seed.require, seed.sha
SELECTION_SHA = "4580b8d7a420ab9f8215fa540fd45110a742839ea34c850f8880781384b4729b"
RANK_SHA = "f1d48798174240a037044d699da8270f6ced2334e068b944e6f9b9ea058b4dfd"


class PinnedSelection:
    """Compile only the immutable collection AST, never a native remesh call."""

    def __init__(self, path):
        self.path = Path(path)
        self.verify()
        tree = ast.parse(self.path.read_text())
        problem = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Problem")
        force = next(node for node in problem.body if isinstance(node, ast.FunctionDef) and node.name == "force_remesh")
        rank = next(node for node in problem.body if isinstance(node, ast.FunctionDef) and node.name == "_rank_independent_template_order")
        start = next(index for index, node in enumerate(force.body) if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name) and node.target.id == "remeshers")
        nodes = force.body[start:start + 2]
        require(len(nodes) == 2 and isinstance(nodes[1], ast.If), "pinned selection layout differs")
        require(hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()
                == SELECTION_SHA, "pinned selection AST differs")
        require(hashlib.sha256(ast.dump(rank, include_attributes=False).encode()).hexdigest() == RANK_SHA,
                "pinned rank-order AST differs")
        function = ast.FunctionDef(name="select_only", args=ast.arguments(posonlyargs=[],
                                  args=[ast.arg(arg="self"), ast.arg(arg="only_domains")],
                                  kwonlyargs=[], kw_defaults=[], defaults=[]),
                                  body=[*copy.deepcopy(nodes), ast.Return(ast.Name(id="remeshers", ctx=ast.Load()))],
                                  decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[function, copy.deepcopy(rank)], type_ignores=[]))
        namespace = {}
        exec(compile(module, str(path) + ":selection-only", "exec"), namespace)  # noqa: S102 - SHA-bound pure collection AST only
        self.selection, self.rank = namespace["select_only"], namespace["_rank_independent_template_order"]

    def verify(self):
        seed.bound_file({"path": str(self.path), "sha256": seed.API["python_problem_sha256"]}, Path.cwd())

    def select(self, problem, requested):
        self.verify()
        class View:
            _rank_independent_template_order = self.rank
        view = View()
        view._meshtemplate_list, view._meshdict = list(problem._meshtemplate_list), dict(problem._meshdict)
        return self.selection(view, requested)


def registration(problem):
    originals, inversion = list(problem._meshtemplate_list), list(problem._domains_remesh_on_inversion)
    domains = sorted(problem._meshdict.items())
    templates = []
    for item in [*originals, *inversion, *(getattr(mesh, "_templatemesh", None) for _, mesh in domains)]:
        if item is not None and not any(item is previous for previous in templates):
            templates.append(item)
    rows = []
    for template in templates:
        remesher = getattr(template, "remesher", None)
        rows.append({"template_id": id(template), "type": type(template).__module__ + "." + type(template).__qualname__,
                     "override_id": id(template._template_override) if getattr(template, "_template_override", None) is not None else None,
                     "owned": template.get_problem() is problem,
                     "remesher_id": id(remesher) if remesher is not None else None,
                     "remesher_template_id": id(remesher.template) if remesher is not None else None,
                     "counter": getattr(remesher, "_cnt", None)})
    return {"original_templates": [id(item) for item in originals],
            "inversion_templates": sorted(id(item) for item in inversion),
            "quality_templates": sorted(id(item) for item in problem._domains_to_remesh),
            "domains": {name: {"mesh_id": id(mesh), "template_id": id(getattr(mesh, "_templatemesh", None))}
                        for name, mesh in domains}, "templates": rows}


def canonicalise(problem, requested, compiler, remesher_type):
    before = registration(problem)
    originals = list(problem._meshtemplate_list)
    selected_templates = originals if requested is None else list(requested)
    require(selected_templates, "empty physical remesh selection")
    raw = compiler.select(problem, requested)
    require(raw and all(type(item) is remesher_type for item in raw), "unknown or absent remesher")
    owner = raw[0]
    require(all(item is owner for item in raw), "distinct physical remeshers requested")
    active = owner.template
    require(active.get_problem() is problem and active.remesher is owner
            and all(active.has_domain(name) for name in ("liquid", "gas")),
            "remesher is unowned or lacks both phases")
    require(all(name in problem._meshdict and problem._meshdict[name]._templatemesh is active
                for name in ("liquid", "gas")),
            "selected remesher template is not active for both phases")
    for template in selected_templates:
        require(template.get_problem() is problem and template.remesher is owner,
                "requested template is unowned or has another remesher")
        require(all(template.has_domain(name) for name in ("liquid", "gas")),
                "requested template lacks both phases")
        require(template is active or any(template is original for original in originals)
                and template._get_template() is active, "requested template is stale or outside the override chain")
    canonical = {active}
    selected = compiler.select(problem, canonical)
    require(len(selected) == 1 and selected[0] is owner,
            "canonical selection does not yield one owned remesher")
    if requested is None:
        require(len(raw) == 1 and raw[0] is owner, "default selection repeats the physical remesher")
    require(before == registration(problem), "canonicalisation changed native registration")
    receipt = {"route": "default" if requested is None else "explicit",
               "requested_template_ids": sorted(id(item) for item in selected_templates),
               "requested_template_types": [type(item).__module__ + "." + type(item).__qualname__ for item in selected_templates],
               "raw_entries": [{"position": index, "remesher_id": id(item), "template_id": id(item.template),
                                "type": type(item).__module__ + "." + type(item).__qualname__, "owned": True,
                                "coverage": {"liquid": True, "gas": True}} for index, item in enumerate(raw)],
               "canonical_template_ids": [id(active)], "remesher_id": id(owner),
               "counter_before": owner._cnt, "base_trunk": owner.base_trunk,
               "selection_ast_sha256": SELECTION_SHA, "rank_order_ast_sha256": RANK_SHA,
               "registration_before": before, "registration_after_selection": registration(problem),
               "registration_changed_by_selection": False}
    return None if requested is None else canonical, owner, receipt


def owned_problem_type(base, hooks_base, compiler, remesher_type, output):
    budget_base = seed.budget_problem_type(base, hooks_base)

    class OwnedRemeshProblem(budget_base):
        def force_remesh(self, only_domains=None, num_adapt=None, interpolator=None):
            phase = self._quality_phase
            require(phase is not None and phase.active, "physical remesh outside the retained phase")
            phase.check_wall()
            delegated, owner, receipt = canonicalise(self, only_domains, compiler, remesher_type)
            receipt["caller"] = inspect.currentframe().f_back.f_code.co_name
            phase.emit("force_remesh_canonicalised", **receipt)
            count, starts, completed = owner._cnt, phase.remesh_starts, phase.remeshes
            hook = phase.current
            converged = hook is not None and hook.converged
            stage = hook.stage_updates if hook is not None else None
            damping = hook.relaxation if hook is not None else None
            directory = Path(output) / "solver/_gmsh"
            before_files = set(directory.glob("*.msh"))
            value = super().force_remesh(delegated, num_adapt, interpolator)
            require(type(value) is bool, "native remesh return type differs")
            if value:
                require(owner._cnt == count + 1 and phase.remesh_starts == starts + 1
                        and phase.remeshes == completed + 1 and not phase.pending_remesh,
                        "one request did not produce one physical generation and hook debit")
                new = set(directory.glob("*.msh")) - before_files
                require(len(new) == 1 and next(iter(new)).name == f"PaperTwoPhaseMesh_REMESH_{count:06d}.msh",
                        "physical remesh generated another mesh or counter skip")
                require(owner.template.get_problem() is self and owner.template.remesher is owner
                        and all(self._meshdict[name]._templatemesh is owner.template
                                for name in ("liquid", "gas")),
                        "remesh lost shared active template ownership")
                if hook is not None and hook.active:
                    require(hook.stage_updates == (0 if converged else stage)
                            and hook.relaxation == (damping if converged else damping * 0.5),
                            "native inversion/quality damping or stage reset semantics changed")
                receipt["mesh"] = {"path": str(next(iter(new))), "sha256": sha(next(iter(new)))}
            else:
                require((owner._cnt, phase.remesh_starts, phase.remeshes) == (count, starts, completed),
                        "false remesh return consumed a physical generation")
            after = registration(self)
            require(after["original_templates"] == receipt["registration_before"]["original_templates"]
                    and after["inversion_templates"] == receipt["registration_before"]["inversion_templates"],
                    "physical remesh changed the retained template registry")
            receipt.update(returned=value, counter_after=owner._cnt, registration_after_remesh=after,
                           stage_updates_before=stage, stage_updates_after=hook.stage_updates if hook is not None else None,
                           relaxation_before=damping, relaxation_after=hook.relaxation if hook is not None else None,
                           post_convergence_quality_remesh=converged)
            if not hasattr(self, "_owned_remesh_receipts"):
                self._owned_remesh_receipts = []
            self._owned_remesh_receipts.append(receipt)
            phase.emit("owned_physical_remesh_verified", **receipt)
            return value

    return OwnedRemeshProblem


def physical_history(problem, ode_type, check):
    """Read all exposed node/internal/position Data histories and pin flags."""
    digest, count = hashlib.sha256(), 0
    def data(label, item):
        nonlocal count
        nvalue, ntime = item.nvalue(), item.ntstorage()
        digest.update(json.dumps([label, nvalue, ntime], separators=(",", ":")).encode())
        for slot in range(nvalue):
            digest.update(struct.pack("<q?", item.eqn_number(slot), item.is_pinned(slot)))
            for time_index in range(ntime):
                digest.update(struct.pack("<d", item.value_at_t(time_index, slot)))
                count += 1
    def walk(mesh, name):
        for index in range(0 if isinstance(mesh, ode_type) else mesh.nnode()):
            node = mesh.node_pt(index)
            data([name, "node", index], node)
            data([name, "position", index], node.variable_position_pt())
            digest.update(struct.pack("<q", node.ndim()))
            for dim in range(node.ndim()):
                digest.update(struct.pack("<d", node.x_lagr(dim)))
            if index % 1024 == 0:
                check()
        for index in range(mesh.nelement()):
            element = mesh.element_pt(index)
            nodes = 0 if isinstance(mesh, ode_type) else element.nnode()
            digest.update(json.dumps([name, "element", index, nodes], separators=(",", ":")).encode())
            for local in range(nodes):
                node = element.node_pt(local)
                digest.update(json.dumps([node.ndim(), [node.eqn_number(slot) for slot in range(node.nvalue())]],
                                         separators=(",", ":")).encode())
                for dim in range(node.ndim()):
                    digest.update(struct.pack("<d", node.x(dim)))
            for slot in range(element.ninternal_data()):
                data([name, "internal", index, slot], element.internal_data_pt(slot))
        for child, interface in sorted(getattr(mesh, "_interfacemeshes", {}).items()):
            walk(interface, name + "/" + child)
    for name, mesh in sorted(problem._meshdict.items()):
        walk(mesh, name)
    return {"data_history_sha256": digest.hexdigest(), "history_values_with_repeats": count}


def pure_read(problem, ode_type, check):
    """Match the accepted probe's read order, retaining the actual flat array."""
    import numpy as np

    dofs, mask = problem.get_current_dofs()
    z, r = problem.interface_polyline()
    flat = np.asarray(problem.get_current_pinned_values(True)).copy()
    arrays = {"free_dofs": dofs, "positional_dof_mask": mask, "pinned_values_with_positions": flat,
              "interface_z": z, "interface_r": r}
    signature = {"arrays": {name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
                                   "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()}
                            for name, value in arrays.items()},
                 "parameters": {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")},
                 "volume": problem.observable("volume", "gas"), "frame_speed": problem.frame_speed(),
                 "time": float(problem.get_current_time(dimensional=False, as_float=True)),
                 "bc_cache_marker": problem._last_bc_setting, "layout": seed.model.stationary_layout(problem, True)}
    measured = physical_history(problem, ode_type, check)
    time = problem.timestepper.time_pt()
    measured.update(signature=signature, pins=seed.diagnostic.actual_pins(problem),
                    native_time_history={"time": float(problem.get_current_time(dimensional=True, as_float=True)),
                                         "dts": [float(time.dt(i)) for i in range(time.ndt())],
                                         "unsteady_steps": problem.timestepper.get_num_unsteady_steps_done(),
                                         "suggested_next_dt": problem._suggested_next_dt,
                                         "writer_endtime": problem._run_statement_endtime,
                                         "loaded_endtime": problem._loaded_run_statement_endtime,
                                         "output_step": problem._output_step},
                    flags={name: getattr(problem, name) for name in seed.FLAGS},
                    settings={name: getattr(problem, name) for name in seed.SETTINGS},
                    steady=[problem.time_stepper_pt(index).is_steady() for index in range(problem.ntime_stepper())])
    return measured, flat
