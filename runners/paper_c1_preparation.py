"""Prove the pinned gas C1 dummy refresh without changing active field storage."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

from runners import run_paper_cold_history as history


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def numeric_npy(raw):
    require(raw[:6] == b'\x93NUMPY', 'unsupported pinned numeric stream or topology')
    major = raw[6]
    count_bytes = 2 if major == 1 else 4
    hlen = int.from_bytes(raw[8:8 + count_bytes], 'little')
    start = 8 + count_bytes
    header = ast.literal_eval(raw[start:start + hlen].decode('latin1' if major < 3 else 'utf8').strip())
    require(header['fortran_order'] is False and header['descr'] in ('<f8', '<i8', '<i4'), 'unsupported pinned numeric stream or topology')
    payload = raw[start + hlen:]
    count = 1
    for dimension in header['shape']:
        count *= dimension
    require(len(payload) == count * {'<f8': 8, '<i8': 8, '<i4': 4}[header['descr']], 'unsupported pinned numeric stream or topology')
    return {'header': header, 'data': payload, 'data_sha256': sha(payload), 'npy_sha256': sha(raw)}

class Stream:

    def __init__(self, raw):
        self.raw = raw
        self.at = history.serialized_metadata(raw)['endtime_offset'] + 8
        self.arrays, self.scalars = ({}, {})

    def take(self, count):
        raw = self.raw[self.at:self.at + count]
        require(len(raw) == count, 'unsupported pinned numeric stream or topology')
        self.at += count
        return raw

    def integer(self):
        return struct.unpack('<q', self.take(8))[0]

    def scalar(self):
        return struct.unpack('<d', self.take(8))[0]

    def string(self):
        count = self.integer()
        require(0 <= count <= 1024 * 1024, 'unsupported pinned numeric stream or topology')
        return self.take(count).decode('ascii')

    def array(self, label):
        start = self.at
        count = self.integer()
        require(0 <= count <= len(self.raw), 'unsupported pinned numeric stream or topology')
        compressed = self.take(count)
        value = numeric_npy(zlib.decompress(compressed))
        self.arrays[label] = {**value, 'start': start, 'end': self.at, 'compressed_bytes': count, 'record_sha256': sha(self.raw[start:self.at])}

    def tracers(self, label):
        count = self.integer()
        self.scalars[label + '/tracer_count'] = count
        require(count == 0, 'unsupported pinned numeric stream or topology')

    def parse(self):
        count = self.integer()
        self.scalars['bulk_mesh_count'] = count
        for _ in range(count):
            mesh = self.string()
            if mesh in ('gas', 'liquid'):
                for name in ('roots', 'sig_lens', 'sig_data', 'node_keys', 'node_lens', 'node_data', 'elem_keys', 'elem_lens', 'elem_data'):
                    self.array(mesh + '/' + name)
                self.tracers(mesh)
            else:
                require(mesh in ('gas_volume', 'globals'), 'unsupported pinned numeric stream or topology')
                fields = self.integer()
                self.scalars[mesh + '/field_count'] = fields
                for _ in range(fields):
                    field = self.string()
                    times = self.integer()
                    self.scalars[mesh + '/' + field] = [self.scalar() for _ in range(times)]
        count = self.integer()
        self.scalars['interface_count'] = count
        for _ in range(count):
            name = self.string()
            for kind in ('keys', 'lens', 'data'):
                self.array('interface/' + name + '/' + kind)
        count = self.integer()
        self.scalars['interface_tracer_mesh_count'] = count
        for _ in range(count):
            self.tracers('interface/' + self.string())
        count = self.integer()
        self.scalars['global_parameter_count'] = count
        for _ in range(count):
            name = self.string()
            self.scalars['parameter/' + name] = self.scalar()
        self.scalars['has_eigendata'] = self.integer()
        require(self.scalars['has_eigendata'] == 1, 'unsupported pinned numeric stream or topology')
        self.scalars['bc_cache_marker'] = self.string()
        self.scalars['eigenpair_count'] = self.integer()
        require(self.scalars['eigenpair_count'] == 0, 'unsupported pinned numeric stream or topology')
        self.scalars['has_continuation'] = self.integer()
        require(self.scalars['has_continuation'] == 0, 'unsupported pinned numeric stream or topology')
        require(self.string() == 'EOF_pyoomph', 'unsupported pinned numeric stream or topology')
        require(self.at == len(self.raw), 'unsupported pinned numeric stream or topology')
        return self

def gas_topology(mesh_path, stream):
    lines = Path(mesh_path).read_text().splitlines()
    def section(name):
        return lines[lines.index("$" + name) + 1:lines.index("$End" + name)]
    require(section("MeshFormat") == ["4.1 0 8"], "unsupported C1 preparation mesh format")
    physical = {}
    for line in section("PhysicalNames")[1:]:
        dim, tag, name = line.split(maxsplit=2)
        physical[(int(dim), int(tag))] = name.strip('"')
    entities = {}
    body = iter(section("Entities"))
    counts = list(map(int, next(body).split()))
    for dim, count in enumerate(counts):
        for _ in range(count):
            values = next(body).split()
            at = 4 if dim == 0 else 7
            entities[(dim, int(values[0]))] = [int(v) for v in values[at + 1:at + 1 + int(values[at])]]
    body = iter(" ".join(section("Nodes")).split())
    blocks, _, _, _ = [int(next(body)) for _ in range(4)]
    coordinates = {}
    for _ in range(blocks):
        _, _, parametric, count = [int(next(body)) for _ in range(4)]
        require(parametric == 0, "parametric C1 topology is unsupported")
        tags = [int(next(body)) for _ in range(count)]
        for tag in tags:
            coordinates[tag] = tuple(float(next(body)) for _ in range(3))
    body = iter(section("Elements"))
    blocks, _, _, _ = map(int, next(body).split())
    elements = []
    for _ in range(blocks):
        dim, entity, kind, count = map(int, next(body).split())
        domain = [physical.get((dim, tag)) for tag in entities[(dim, entity)]]
        for _ in range(count):
            row = list(map(int, next(body).split()))
            if dim == 2 and "gas" in domain:
                require(kind == 9 and len(row) == 7 and domain == ["gas"], "ambiguous gas triangle topology")
                nodes = row[1:]
                area = sum(coordinates[nodes[k]][0] * coordinates[nodes[(k + 1) % 3]][1]
                           - coordinates[nodes[(k + 1) % 3]][0] * coordinates[nodes[k]][1] for k in range(3))
                require(math.isfinite(area) and area != 0, "degenerate gas triangle")
                if area < 0:
                    nodes = [nodes[i] for i in (0, 2, 1, 5, 4, 3)]
                elements.append(nodes)
    require(elements, "gas topology is absent")
    expected, corners, masters = {}, set(), {}
    for root, nodes in enumerate(elements):
        for local, node in enumerate(nodes):
            expected[node] = min(expected.get(node, (root, 1, local)), (root, 1, local))
        corners.update(nodes[:3])
        for local, (a, b) in {3: (0, 1), 4: (1, 2), 5: (0, 2)}.items():
            pair = tuple(sorted((nodes[a], nodes[b])))
            require(nodes[local] not in masters or masters[nodes[local]] == pair, "ambiguous gas midnode masters")
            masters[nodes[local]] = pair
    require(not corners.intersection(masters), "gas midnode is also an active corner")
    keys = [tuple(v) for v in struct.iter_unpack("<qqq", stream.arrays["gas/node_keys"]["data"])]
    roots = [v[0] for v in struct.iter_unpack("<q", stream.arrays["gas/roots"]["data"])]
    element_keys = [tuple(v) for v in struct.iter_unpack("<qq", stream.arrays["gas/elem_keys"]["data"])]
    require(roots == list(range(len(elements))) and element_keys == [(r, 1) for r in roots],
            "refined or reordered gas topology is unsupported")
    require(len(set(keys)) == len(keys) and set(keys) == set(expected.values()), "gas structural keys differ from mesh")
    key_rows = {key: row for row, key in enumerate(keys)}
    node_rows = {node: key_rows[key] for node, key in expected.items()}
    return {"keys": keys, "corner_rows": sorted(node_rows[n] for n in corners),
            "mid_rows": {node_rows[n]: tuple(node_rows[m] for m in pair) for n, pair in masters.items()},
            "triangles": len(elements)}


def verify_gas_preparation(before, prepared, mesh_path):
    """Require an exact half-sum at every gas midnode and storage level."""
    left, right = Stream(before).parse(), Stream(prepared).parse()
    require(left.scalars == right.scalars and left.arrays.keys() == right.arrays.keys(),
            "preparation changed scalar or record structure")
    a, b = left.arrays["gas/node_data"], right.arrays["gas/node_data"]
    require(before[:a["start"]] == prepared[:b["start"]] and before[a["end"]:] == prepared[b["end"]:],
            "preparation changed bytes outside gas node_data")
    require(a["header"] == b["header"] and a["header"]["descr"] == "<f8",
            "preparation changed gas node_data layout")
    topology = gas_topology(mesh_path, left)
    lengths = [v[0] for v in struct.iter_unpack("<i", left.arrays["gas/node_lens"]["data"])]
    require(len(lengths) == len(topology["keys"]) and all(length == 44 for length in lengths),
            "unknown gas pressure or storage layout")
    require(len(a["data"]) == len(b["data"]) == 44 * 8 * len(lengths), "gas data length differs")
    midrows = topology["mid_rows"]
    changed = []
    changed_nodes = 0
    for row in range(len(lengths)):
        for slot in range(44):
            index = row * 44 + slot
            x, y = a["data"][index * 8:(index + 1) * 8], b["data"][index * 8:(index + 1) * 8]
            if x != y:
                require(row in midrows and 37 <= slot < 44, "preparation changed an active, history, geometry or other-field slot")
                changed.append({"node_key": topology["keys"][row], "storage_level": slot - 37,
                                "before_hex": x.hex(), "prepared_hex": y.hex()})
    for row, masters in midrows.items():
        modified = False
        for t in range(7):
            index = row * 44 + 37 + t
            values = [struct.unpack_from("<d", a["data"], (m * 44 + 37 + t) * 8)[0] for m in masters]
            require(all(math.isfinite(v) for v in values), "nonfinite pressure corner")
            expected = struct.pack("<d", (values[0] + values[1]) / 2.0)
            actual = b["data"][index * 8:(index + 1) * 8]
            require(actual == expected, "gas dummy half-sum differs bit-for-bit")
            modified |= a["data"][index * 8:(index + 1) * 8] != actual
        changed_nodes += int(modified)
    proof = {"kind": "gas-c1-dummy-preparation", "operation": "first existing pure_read gas volume observable",
             "meshes": ["gas"], "verified": True, "pressure_nodal_offset": 3, "storage_levels": 7,
             "before_sha256": sha(before), "prepared_sha256": sha(prepared),
             "mesh_sha256": sha(Path(mesh_path).read_bytes()), "triangles": topology["triangles"],
             "gas_midnodes": len(midrows), "half_sum_values_verified": len(midrows) * 7,
             "changed_midnodes": changed_nodes, "unchanged_midnodes": len(midrows) - changed_nodes,
             "active_corner_storage_unchanged": True, "all_other_field_geometry_history_unchanged": True,
             "all_non_gas_records_scalars_prefix_suffix_unchanged": True,
             "topology_unambiguous": True, "actual_changes": changed}
    return proof, topology


def flags_record(problem, ode_type):
    """Read equation numbers and pin flags only; no field or observable getter."""
    digest = hashlib.sha256()
    def data(label, item):
        digest.update(json.dumps([label, item.nvalue()], separators=(",", ":")).encode())
        for slot in range(item.nvalue()):
            digest.update(struct.pack("<q?", item.eqn_number(slot), item.is_pinned(slot)))
    def walk(mesh, name):
        for index in range(0 if isinstance(mesh, ode_type) else mesh.nnode()):
            node = mesh.node_pt(index)
            data([name, "node", index], node)
            data([name, "position", index], node.variable_position_pt())
        for index in range(mesh.nelement()):
            element = mesh.element_pt(index)
            for slot in range(element.ninternal_data()):
                data([name, "internal", index, slot], element.internal_data_pt(slot))
        for child, interface in sorted(getattr(mesh, "_interfacemeshes", {}).items()):
            walk(interface, name + "/" + child)
    for name, mesh in sorted(problem._meshdict.items()):
        walk(mesh, name)
    return {"all_equation_numbers_and_pin_flags_sha256": digest.hexdigest()}


def prove_inactive_gas_mids(problem, topology):
    """Prove the enumerated midnode pressure slots are pinned and unnumbered."""
    mesh = problem._meshdict["gas"]
    element_keys = list(mesh.get_element_structural_keys())
    indices, stride = mesh.get_element_node_indices()
    require(stride == 6, "native gas elements are not quadratic triangles")
    native_keys = {}
    for row in range(mesh.nelement()):
        root, path = element_keys[row * 2:row * 2 + 2]
        for local in range(6):
            node = int(indices[row * stride + local])
            key = (int(root), int(path), local)
            native_keys[node] = min(native_keys.get(node, key), key)
    require(set(native_keys.values()) == set(topology["keys"]), "native gas node topology differs")
    key_indices = {key: node for node, key in native_keys.items()}
    for row in topology["mid_rows"]:
        node = mesh.node_pt(key_indices[topology["keys"][row]])
        require(node.nvalue() == 4 and node.ntstorage() == 7 and node.is_pinned(3)
                and node.eqn_number(3) < 0, "gas dummy pressure is active or has another layout")
    return {"verified": True, "pinned_unassigned_pressure_midnodes": len(topology["mid_rows"])}
