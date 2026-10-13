"""Independent, numeric-only reader for the pinned global native state stream.

Reading bytes establishes their structure, not native load or solver validity.
No pickle, pyoomph import, mesh loading or native array routine is used.
"""

from __future__ import annotations

import ast
import hashlib
import io
import itertools
import math
import struct
import zlib
from dataclasses import dataclass


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 256 * 1024 * 1024
    max_items: int = 16 * 1024 * 1024
    max_records: int = 1024 * 1024
    max_string: int = 1024 * 1024
    max_header: int = 64 * 1024


@dataclass(frozen=True)
class NumericArray:
    dtype: str
    shape: tuple[int, ...]
    values: tuple
    packed: bytes  # Canonical C-order bytes; retain IEEE payloads exactly.

    def description(self):
        return {"dtype": self.dtype, "shape": list(self.shape),
                "sha256": hashlib.sha256(self.packed).hexdigest()}


class Reader:
    def __init__(self, data, limits):
        require(isinstance(data, bytes) and len(data) <= limits.max_bytes, "state byte bound exceeded")
        self.stream, self.limits, self.compression = io.BytesIO(data), limits, None
        self.array_bytes = 0

    def read(self, count):
        require(type(count) is int and 0 <= count <= self.limits.max_bytes, "invalid byte length")
        value = self.stream.read(count)
        require(len(value) == count, "truncated native stream")
        return value

    def integer(self):
        return struct.unpack("<q", self.read(8))[0]

    def count(self, label):
        value = self.integer()
        require(0 <= value <= self.limits.max_records, "invalid " + label + " count")
        return value

    def flag(self, label):
        value = self.integer()
        require(value in (0, 1), "invalid " + label + " flag")
        return value

    def scalar(self, *, nullable=False):
        value = struct.unpack("<d", self.read(8))[0]
        if nullable and math.isnan(value):
            return None
        require(math.isfinite(value), "nonfinite native scalar")
        return value

    def string(self):
        length = self.integer()
        require(0 <= length <= self.limits.max_string, "invalid native string length")
        value = self.read(length).decode("ascii")
        require("\0" not in value, "native string contains NUL")
        return value

    def array(self):
        if self.compression is None:
            result = self.npy(self.stream)
            require(result.shape and result.shape[0] == 1, "missing uncompressed array wrapper")
            result = NumericArray(result.dtype, result.shape[1:], result.values, result.packed)
        else:
            length = self.integer()
            require(0 <= length <= self.limits.max_bytes, "invalid compressed array length")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(self.read(length), self.limits.max_bytes + 1)
            require(len(raw) <= self.limits.max_bytes and decoder.eof
                    and not decoder.unused_data and not decoder.unconsumed_tail,
                    "compressed array exceeds bounds or contains trailing data")
            stream = io.BytesIO(raw)
            result = self.npy(stream)
            require(stream.read(1) == b"", "trailing NPY bytes")
        self.array_bytes += len(result.packed)
        require(self.array_bytes <= self.limits.max_bytes, "cumulative numeric payload bound exceeded")
        return result

    def npy(self, stream):
        def read(n):
            value = stream.read(n)
            require(len(value) == n, "truncated NPY array")
            return value

        require(read(6) == b"\x93NUMPY", "invalid NPY magic")
        version = tuple(read(2))
        require(version in {(1, 0), (2, 0), (3, 0)}, "unsupported NPY version")
        length = struct.unpack("<H" if version == (1, 0) else "<I", read(2 if version == (1, 0) else 4))[0]
        require(length <= self.limits.max_header, "NPY header bound exceeded")
        expression = ast.parse(read(length).decode("utf-8" if version == (3, 0) else "latin1"), mode="eval")
        require(isinstance(expression.body, ast.Dict), "NPY header must be a literal dictionary")
        keys = [ast.literal_eval(key) for key in expression.body.keys]
        require(all(isinstance(key, str) for key in keys) and len(set(keys)) == len(keys), "duplicate/invalid NPY header key")
        header = ast.literal_eval(expression)
        require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
                "unsupported NPY header")
        codecs = {"<f8": ("d", 8), "<i8": ("q", 8), "<i4": ("i", 4), "|u1": ("B", 1),
                  "|b1": ("?", 1), "<c16": ("dd", 16)}
        require(isinstance(header["descr"], str) and header["descr"] in codecs,
                "object, structured or unsupported numeric dtype")
        shape = header["shape"]
        require(isinstance(shape, tuple) and len(shape) <= 8
                and all(type(n) is int and n >= 0 for n in shape)
                and type(header["fortran_order"]) is bool, "invalid NPY shape/order")
        count = math.prod(shape)
        codec, size = codecs[header["descr"]]
        require(count <= self.limits.max_items and count * size <= self.limits.max_bytes,
                "NPY item/byte bound exceeded")
        packed = read(count * size)
        if header["fortran_order"] and len(shape) > 1:
            strides = [math.prod(shape[:i]) for i in range(len(shape))]
            offsets = (sum(i * s for i, s in zip(index, strides, strict=True))
                       for index in itertools.product(*(range(n) for n in shape)))
            packed = b"".join(packed[i * size:(i + 1) * size] for i in offsets)
        values = tuple(struct.iter_unpack("<" + codec, packed))
        values = tuple(v[0] if len(v) == 1 else v for v in values)
        return NumericArray(header["descr"], shape, values, packed)


def records(keys, lengths, values, width, label):
    require(keys.dtype == "<i8" and len(keys.shape) == 2 and keys.shape[1] == width,
            "invalid " + label + " keys")
    require(lengths.dtype in {"<i4", "<i8"} and lengths.shape == (keys.shape[0],)
            and values.dtype == "<f8" and len(values.shape) == 1, "invalid " + label + " records")
    require(all(n >= 0 for n in lengths.values) and sum(lengths.values) == len(values.values),
            "invalid " + label + " block lengths")
    names = [tuple(keys.values[i:i + width]) for i in range(0, len(keys.values), width)]
    require(len(set(names)) == len(names), "duplicate " + label + " keys")
    return {"keys": keys, "lengths": lengths, "data": values}


def decode(data, *, limits=None):
    """Read every section of format 0.1.6; reject unsupported tracer payloads."""
    reader = Reader(data, Limits() if limits is None else limits)
    require((reader.string(), reader.string(), reader.string()) == ("pyoomph_dump", "0.1.6", "global"),
            "unsupported native state format")
    result = {"time": reader.scalar(), "output_step": reader.count("output step"),
              "continue_step": reader.count("continue step")}
    compression = reader.integer()
    require(compression == -100 or 0 <= compression <= 9, "unsupported compression")
    reader.compression = None if compression == -100 else compression
    result["compression"] = reader.compression
    templates = []
    for _ in range(reader.count("template")):
        start = reader.stream.tell()
        name = reader.string()
        end = reader.stream.tell()
        remesher = reader.flag("remesher")
        templates.append({"stored_path": name, "string_range": [start, end],
                          "remesher_counter": reader.count("remesher") if remesher else None})
    require(len(templates) == 1 and templates[0]["stored_path"].endswith(".msh"),
            "selected model requires one bound mesh template")
    result["templates"] = templates
    ndt = reader.count("dt")
    require(ndt > 0, "empty native dt vector")
    result["dts"] = [reader.scalar() for _ in range(ndt)]
    require(all(value > 0 for value in result["dts"]), "nonpositive native dt")
    result.update(counter=reader.count("unsteady step"), suggested_next_dt=reader.scalar(nullable=True),
                  run_endtime=reader.scalar(nullable=True))
    require(result["suggested_next_dt"] is None or result["suggested_next_dt"] > 0,
            "invalid suggested next dt")
    meshes = {}
    expected = {"gas", "liquid", "globals", "gas_volume"}
    require(reader.count("mesh") == len(expected), "selected model mesh count differs")
    for name_expected in sorted(expected):
        name = reader.string()
        require(name == name_expected, "native mesh order/identity differs")
        if name in {"gas_volume", "globals"}:
            fields = {}
            for _ in range(reader.count("ODE field")):
                field = reader.string()
                require(field and field not in fields, "duplicate/empty ODE field")
                nt = reader.count("ODE history")
                require(nt in (5, 7), "unsupported ODE history layout")
                fields[field] = [reader.scalar() for _ in range(nt)]
            require(list(fields) == sorted(fields), "ODE field order differs")
            meshes[name] = {"kind": "ode", "fields": fields}
        else:
            arrays = dict(zip(("roots", "sig_lens", "sig_data", "node_keys", "node_lens", "node_data",
                               "elem_keys", "elem_lens", "elem_data"),
                              (reader.array() for _ in range(9)), strict=True))
            require(arrays["roots"].dtype == "<i8" and len(arrays["roots"].shape) == 1
                    and arrays["sig_lens"].dtype in {"<i4", "<i8"}
                    and arrays["sig_lens"].shape == arrays["roots"].shape
                    and arrays["sig_data"].dtype in {"<i4", "<i8"}
                    and len(arrays["sig_data"].shape) == 1
                    and sum(arrays["sig_lens"].values) == len(arrays["sig_data"].values),
                    "invalid refinement signature")
            records(arrays["node_keys"], arrays["node_lens"], arrays["node_data"], 3, "node")
            records(arrays["elem_keys"], arrays["elem_lens"], arrays["elem_data"], 2, "element")
            require(reader.count("bulk tracer") == 0, "tracer collections are outside the selected model")
            meshes[name] = {"kind": "spatial", "arrays": arrays}
    result["meshes"] = meshes
    interfaces = {}
    for _ in range(reader.count("interface")):
        name = reader.string()
        require(name and name not in interfaces, "duplicate/empty interface name")
        interfaces[name] = records(reader.array(), reader.array(), reader.array(), 3, "interface")
    require(list(interfaces) == sorted(interfaces), "interface order differs")
    result["interfaces"] = interfaces
    names = []
    for _ in range(reader.count("interface tracer mesh")):
        name = reader.string()
        require(name in interfaces and name not in names, "unknown/duplicate interface tracer mesh")
        names.append(name)
        require(reader.count("interface tracer") == 0, "interface tracers are outside the selected model")
    require(names == sorted(interfaces), "interface tracer inventory differs")
    parameters = {}
    for _ in range(reader.count("global parameter")):
        name = reader.string()
        require(name and name not in parameters, "duplicate/empty global parameter")
        parameters[name] = reader.scalar()
    require(list(parameters) == sorted(parameters), "global parameter order differs")
    result["parameters"] = parameters
    result["eigen_section_start"] = reader.stream.tell()
    eigen = {}
    if reader.flag("eigen data"):
        eigen["bc_cache"] = reader.string()
        count = reader.count("eigenpair")
        eigen["count"] = count
        if count:
            eigen["values"], eigen["vectors"] = reader.array(), reader.array()
            require(eigen["values"].shape == (count,) and len(eigen["vectors"].shape) == 2
                    and eigen["vectors"].shape[0] == count, "invalid eigen array shape")
            if reader.flag("azimuthal eigen data"):
                eigen["azimuthal"] = reader.array()
                require(eigen["azimuthal"].shape == (count,), "invalid azimuthal array shape")
    result["eigen"] = eigen
    continuation = {}
    if reader.flag("continuation"):
        continuation = {"direction": reader.array(), "parameter_derivative": reader.scalar(),
                        "theta_squared": reader.scalar()}
    result["continuation"] = continuation
    require(reader.string() == "EOF_pyoomph" and reader.stream.read(1) == b"", "invalid native footer/trailing bytes")
    result["sha256"] = hashlib.sha256(data).hexdigest()
    return result


def portable_bytes(data, decoded, mesh_name):
    """Replace only the proved template string; leave all numeric bytes intact."""
    require(decoded["sha256"] == hashlib.sha256(data).hexdigest(), "decoder belongs to another state")
    require(isinstance(mesh_name, str) and mesh_name.endswith(".msh")
            and "/" not in mesh_name and "\\" not in mesh_name and "\0" not in mesh_name,
            "portable mesh must be one local regular filename")
    start, end = decoded["templates"][0]["string_range"]
    encoded = mesh_name.encode("ascii")
    return data[:start] + struct.pack("<q", len(encoded)) + encoded + data[end:]
