"""Exact gas-only preparation predicates on retained bytes and synthetic faults."""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from runners import paper_c1_preparation as c1


@pytest.fixture
def evidence():
    names = ("STONE_C1_BEFORE", "STONE_C1_PREPARED", "STONE_C1_MESH")
    if not all(os.environ.get(name) for name in names):
        pytest.skip("retained-byte tests require external C1 evidence paths")
    before, prepared, mesh = (Path(os.environ[name]) for name in names)
    assert not any(name.startswith("pyoomph") for name in sys.modules)
    return before.read_bytes(), prepared.read_bytes(), mesh


def change_array(raw, label, byte_offset):
    stream = c1.Stream(raw).parse()
    record = stream.arrays[label]
    npy = bytearray(zlib.decompress(raw[record["start"] + 8:record["end"]]))
    at = len(npy) - len(record["data"]) + byte_offset
    npy[at] ^= 1
    compressed = zlib.compress(npy, level=6)
    return raw[:record["start"]] + struct.pack("<q", len(compressed)) + compressed + raw[record["end"]:]


def test_exact_retained_gas_half_sums_and_unchanged_liquid(evidence):
    before, prepared, mesh = evidence
    proof, topology = c1.verify_gas_preparation(before, prepared, mesh)
    assert proof["verified"] is True and proof["meshes"] == ["gas"]
    assert proof["half_sum_values_verified"] == 82257
    assert (proof["changed_midnodes"], proof["unchanged_midnodes"]) == (11399, 352)
    assert len(proof["actual_changes"]) == 79793
    assert len(topology["corner_rows"]) == 4072


@pytest.mark.parametrize("fault", ["active_corner", "other_field", "geometry", "history", "bad_average", "liquid"])
def test_actual_predicates_refuse_single_bit_faults(evidence, fault):
    before, prepared, mesh = evidence
    stream = c1.Stream(before).parse()
    topology = c1.gas_topology(mesh, stream)
    corner = topology["corner_rows"][0]
    mid = next(iter(topology["mid_rows"]))
    if fault == "history":
        offset = c1.history.serialized_metadata(prepared)["endtime_offset"] - 32
        damaged = bytearray(prepared)
        damaged[offset] ^= 1
        damaged = bytes(damaged)
    else:
        label = "liquid/node_data" if fault == "liquid" else "gas/node_data"
        index = {"active_corner": corner * 44 + 37, "other_field": mid * 44 + 16,
                 "geometry": mid * 44, "bad_average": mid * 44 + 40, "liquid": 0}[fault]
        damaged = change_array(prepared, label, index * 8)
    with pytest.raises(ValueError):
        c1.verify_gas_preparation(before, damaged, mesh)


def test_ambiguous_midnode_topology_refuses(evidence, tmp_path):
    before, prepared, mesh = evidence
    lines = mesh.read_text().splitlines()
    start = lines.index("$Elements") + 2
    blocks = int(lines[start - 1].split()[0])
    at = start
    for _ in range(blocks):
        dim, entity, kind, count = map(int, lines[at].split())
        if dim == 2 and entity == 2 and kind == 9:
            row = lines[at + 1].split()
            row[4] = row[1]
            lines[at + 1] = " ".join(row)
            break
        at += count + 1
    else:
        raise AssertionError("gas triangle block missing")
    damaged = tmp_path / "ambiguous.msh"
    damaged.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="midnode|ambiguous|structural"):
        c1.verify_gas_preparation(before, prepared, damaged)


@pytest.mark.parametrize("active", [False, True])
def test_native_inactivity_contract_without_native_import(active):
    topology = {"keys": [(0, 1, i) for i in range(6)],
                "mid_rows": {3: (0, 1), 4: (1, 2), 5: (0, 2)}}
    node = SimpleNamespace(nvalue=lambda: 4, ntstorage=lambda: 7,
                           is_pinned=lambda slot: not active, eqn_number=lambda slot: 0 if active else -1)
    mesh = SimpleNamespace(nelement=lambda: 1, get_element_structural_keys=lambda: [0, 1],
                           get_element_node_indices=lambda: ([0, 1, 2, 3, 4, 5], 6), node_pt=lambda index: node)
    problem = SimpleNamespace(_meshdict={"gas": mesh})
    if active:
        with pytest.raises(ValueError, match="pressure is active"):
            c1.prove_inactive_gas_mids(problem, topology)
    else:
        assert c1.prove_inactive_gas_mids(problem, topology)["pinned_unassigned_pressure_midnodes"] == 3


def test_no_pickle_or_native_array_decoder(evidence):
    before, prepared, mesh = evidence
    proof, _ = c1.verify_gas_preparation(before, prepared, mesh)
    json.dumps(proof, allow_nan=False)
    assert not any(name.startswith("pyoomph") for name in sys.modules)
