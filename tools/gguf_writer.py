"""A minimal GGUF v3 writer: metadata of every type and F32 tensors, no ggml and no gguf-py.

The tests write small GGUF files with it (a tokenizer's metadata, a shard with no tokenizer) and read them back with
`tools/gguf_reader.py`, the reader the server uses.  gguf-py is not among Strata's requirements, so this stays in
the standard library plus numpy.  It writes the format only; it does not quantize.

Self-check: `python tools/gguf_writer.py` writes a file, reads it back with `gguf_reader.py`, and checks that the
metadata and the tensors survived byte for byte.
"""
from __future__ import annotations

import dataclasses
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUF_MAGIC, GGUFFile  # noqa: E402

# ggml_type ids we can write. Values match GGML_TYPES in gguf_reader.py.
TYPE_IDS = {"F32": 0}
DEFAULT_ALIGNMENT = 32

_META_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
             "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12}


# ------------------------------------------------------------------ writer
@dataclasses.dataclass
class _Tensor:
    name: str
    shape: list[int]          # GGUF order: dim 0 varies fastest
    type_name: str
    data: bytes


class GGUFWriter:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.metadata: dict[str, tuple[str, object]] = {}
        self.tensors: list[_Tensor] = []

    # ---- metadata
    def add(self, key: str, value, type_name: str | None = None) -> None:
        if type_name is None:
            type_name = {bool: "bool", int: "i64", float: "f64", str: "string"}.get(type(value))
        if type_name is None:
            if isinstance(value, list) and value:
                elem = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]))
                if elem is None:
                    raise TypeError(f"unsupported array element {type(value[0])} for {key}")
                type_name = "array:" + elem
            else:
                raise TypeError(f"cannot infer a GGUF type for {key}")
        self.metadata[key] = (type_name, value)

    # ---- tensors
    def add_f32(self, name: str, arr: np.ndarray) -> None:
        """A row-major array; its GGUF shape is numpy's reversed (the last axis varies fastest)."""
        a = np.ascontiguousarray(arr, dtype="<f4")
        self.tensors.append(_Tensor(name, list(a.shape[::-1]), "F32", a.tobytes()))

    # ---- serialisation
    def _kv_bytes(self, key: str, type_name: str, value) -> bytes:
        out = bytearray()
        out += struct.pack("<Q", len(key)) + key.encode("utf-8")
        if type_name.startswith("array:"):
            elem = type_name.split(":", 1)[1]
            out += struct.pack("<I", _META_IDS["array"])
            out += struct.pack("<I", _META_IDS[elem])
            out += struct.pack("<Q", len(value))
            for v in value:
                out += self._scalar(elem, v)
            return bytes(out)
        out += struct.pack("<I", _META_IDS[type_name])
        out += self._scalar(type_name, value)
        return bytes(out)

    @staticmethod
    def _scalar(type_name: str, v) -> bytes:
        if type_name == "string":
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
               "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[type_name]
        return struct.pack(fmt, v)

    def write(self, path) -> pathlib.Path:
        path = pathlib.Path(path)
        body = bytearray()
        body += struct.pack("<I", GGUF_MAGIC)
        body += struct.pack("<I", 3)                       # GGUF v3
        body += struct.pack("<Q", len(self.tensors))
        body += struct.pack("<Q", len(self.metadata))
        for k, (t, v) in self.metadata.items():
            body += self._kv_bytes(k, t, v)

        # Tensor directory: offsets are relative to the start of the data section.
        offset = 0
        infos = bytearray()
        for t in self.tensors:
            infos += struct.pack("<Q", len(t.name)) + t.name.encode("utf-8")
            infos += struct.pack("<I", len(t.shape))
            infos += struct.pack(f"<{len(t.shape)}Q", *t.shape)
            infos += struct.pack("<I", TYPE_IDS[t.type_name])
            infos += struct.pack("<Q", offset)
            offset += len(t.data)
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment
        body += infos

        pad = (-len(body)) % self.alignment
        body += b"\0" * pad
        for t in self.tensors:
            body += t.data
            pad = (-len(t.data)) % self.alignment
            body += b"\0" * pad

        path.write_bytes(bytes(body))
        return path


# ------------------------------------------------------------------ self-check
def _selfcheck() -> int:
    rng = np.random.default_rng(7)
    w = GGUFWriter()
    w.add("general.architecture", "test")
    w.add("test.block_count", 8, "u32")
    w.add("test.array", [1, 2, 3, 4], "array:i32")
    w.add("general.name", "strata tiny")

    A = rng.standard_normal((16, 64)).astype(np.float32)
    B = rng.standard_normal((4, 8)).astype(np.float32)
    w.add_f32("blk.0.a", A)
    w.add_f32("blk.0.b", B)

    with tempfile.TemporaryDirectory() as d:
        p = w.write(pathlib.Path(d) / "selftest.gguf")
        print(f"wrote {p.name}  ({p.stat().st_size} B)")
        g = GGUFFile(p)
        ok = True
        print(f"  version {g.version}  tensors {len(g.tensors)}  metadata {len(g.metadata)}")
        if g.metadata.get("general.architecture") != "test": print("  FAIL arch"); ok = False
        if g.metadata.get("test.block_count") != 8:         print("  FAIL u32"); ok = False
        if g.metadata.get("test.array") != [1, 2, 3, 4]:    print("  FAIL array"); ok = False

        by = {t.name: t for t in g.tensors}
        for name, arr in (("blk.0.a", A), ("blk.0.b", B)):
            t = by.get(name)
            if t is None or t.type_name != "F32" or t.shape != list(arr.shape[::-1]):
                print(f"  FAIL tensor dir: {t}"); ok = False
                continue
            with p.open("rb") as fh:
                fh.seek(g.data_start + t.offset)
                raw = fh.read(arr.size * 4)
            if raw != arr.astype("<f4").tobytes():
                print(f"  FAIL {name}: the data differs"); ok = False
    print("SELF-CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
