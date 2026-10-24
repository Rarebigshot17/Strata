#!/usr/bin/env python3
"""How busy each card was in an Nsight Systems profile: computing, copying, both, or idle.

    nsys profile -o prefill --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx \\
        env STRATA_GLM_PROFILE=FROM,N build/strata-glm ...
    nsys export --type sqlite -o prefill.sqlite prefill.nsys-rep
    python3 tools/nsys_busy.py prefill.sqlite

Per card: the time its kernels ran (overlapping kernels counted once), the time its copies from RAM, to RAM and
between cards ran, how much of the copying from RAM overlapped its kernels, and the rate of that copying. Then how
long 0, 1, 2 or 3 cards were computing at once, the prompt chunks (the engine's NVTX ranges "chunk POS+T"), each
card's kernel time by the part of the engine that launched it (the innermost NVTX range open on the launching thread,
found through the launch call's correlation id: a layer's attention half, its routed experts, another card's experts
on a helper thread, the rest of its FFN half), the host's calls into the CUDA runtime by count and time, and the
kernels that took the most time on each card. Standard library only. STRATA_GLM_PROFILE_EVAL=FROM,N records N
decoded tokens instead of prompt chunks.
"""
from __future__ import annotations

import argparse
import bisect
import re
import sqlite3
import sys
from collections import defaultdict

COPY_KINDS = {1: "RAM->card", 2: "card->RAM", 8: "card->card", 10: "card->card (peer)"}


def merge(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, non-overlapping intervals covering the same time."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def total(iv: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in iv)


def intersect(x: list[tuple[int, int]], y: list[tuple[int, int]]) -> int:
    """Time covered by both merged interval lists."""
    i = j = t = 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            t += b - a
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return t


def columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def has_table(db: sqlite3.Connection, table: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def strings(db: sqlite3.Connection) -> dict[int, str]:
    if not has_table(db, "StringIds"):
        return {}
    return {int(i): v for i, v in db.execute("SELECT id, value FROM StringIds")}


def name_of(v, ids: dict[int, str]) -> str:
    if isinstance(v, int):
        return ids.get(v, str(v))
    return str(v)


PHASES = ("attention half", "routed experts, its own layer", "routed experts, another card's layer",
          "FFN half, the rest", "other")


def phase_of(name: str | None, main_thread: bool) -> str:
    """The part of the engine a kernel belongs to, from the innermost NVTX range open where it was launched."""
    if name is None:
        return "other" if main_thread else "routed experts, another card's layer"
    if re.fullmatch(r"L\d+ (MLA|KDA)", name):
        return "attention half"
    if name.startswith("routed experts: queue") or name == "CPU experts":
        # the prompt path's own experts; in a decoded token, the cached experts launched while the CPU computes
        return "routed experts, its own layer"
    if re.fullmatch(r"L\d+ FFN", name) or name.startswith("routing:"):
        return "FFN half, the rest"
    return "other"


def kernel_phases(db: sqlite3.Connection, ids: dict[int, str]) -> dict[int, dict[str, list[tuple[int, int]]]] | None:
    """Per card, its kernels' intervals by phase_of; None when the profile lacks the tables it needs."""
    need = {"CUPTI_ACTIVITY_KIND_KERNEL": {"correlationId"},
            "CUPTI_ACTIVITY_KIND_RUNTIME": {"correlationId", "globalTid"}, "NVTX_EVENTS": {"globalTid"}}
    for t, cols in need.items():
        if not has_table(db, t) or not cols <= columns(db, t):
            return None
    launch = {int(c): (int(tid), int(st)) for c, tid, st in
              db.execute("SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    ncols = columns(db, "NVTX_EVENTS")
    txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in ncols else "text"
    ranges: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for st, en, t, tid in db.execute(f"SELECT start, end, {txt}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        if t is not None and tid is not None:
            ranges[int(tid)].append((int(st), int(en), str(t)))
    starts = {}
    for tid in ranges:
        ranges[tid].sort(key=lambda r: (r[0], -r[1]))   # of ranges that start together, the outer one first
        starts[tid] = [r[0] for r in ranges[tid]]
    # the thread that runs the chunks is the main one; others launching kernels are the helpers'
    main_tids = {tid for tid, rs in ranges.items() if any(r[2].startswith("chunk ") for r in rs)}

    def innermost(tid: int, t: int) -> str | None:
        rs = ranges.get(tid)
        if not rs:
            return None
        j = bisect.bisect_right(starts[tid], t) - 1
        for _ in range(4000):                           # the latest-started range still open at t
            if j < 0:
                return None
            if rs[j][1] >= t:
                return rs[j][2]
            j -= 1
        return None

    out: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for s, e, dev, corr in db.execute("SELECT start, end, deviceId, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        tid, at = launch.get(int(corr), (None, None))
        name = innermost(tid, at) if tid is not None else None
        out[dev][phase_of(name, tid in main_tids)].append((s, e))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sqlite", help="the profile exported with `nsys export --type sqlite`")
    ap.add_argument("--top", type=int, default=12, help="kernels listed per card")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    ids = strings(db)

    kern: dict[int, list[tuple[int, int]]] = defaultdict(list)
    kern_by_name: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if has_table(db, "CUPTI_ACTIVITY_KIND_KERNEL"):
        cols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        nm = "shortName" if "shortName" in cols else "demangledName"
        for s, e, dev, n in db.execute(f"SELECT start, end, deviceId, {nm} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            kern[dev].append((s, e))
            k = kern_by_name[dev][name_of(n, ids)]
            k[0] += 1
            k[1] += e - s
    copies: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    copy_bytes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for s, e, dev, kind, nbytes in db.execute(
                "SELECT start, end, deviceId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            what = COPY_KINDS.get(kind, f"kind {kind}")
            copies[dev][what].append((s, e))
            copy_bytes[dev][what] += nbytes
    devices = sorted(set(kern) | set(copies))
    if not devices:
        print("no kernels or copies in this profile (was it captured? see --capture-range)")
        return 1
    # the activity tables' deviceId is the process's CUDA device number; the GPU table numbers the GPUs its own way,
    # so a name is only taken through a CUDA number (TARGET_INFO_GPU.cuDevice, or TARGET_INFO_CUDA_DEVICE's
    # cudaId -> gpuId), else none is shown
    names: dict[int, str] = {}
    if has_table(db, "TARGET_INFO_GPU") and {"cuDevice", "name"} <= columns(db, "TARGET_INFO_GPU"):
        for cu, n in db.execute("SELECT cuDevice, name FROM TARGET_INFO_GPU"):
            if cu is not None:
                names.setdefault(int(cu), name_of(n, ids))
    if (not names and has_table(db, "TARGET_INFO_CUDA_DEVICE") and has_table(db, "TARGET_INFO_GPU")
            and {"cudaId", "gpuId"} <= columns(db, "TARGET_INFO_CUDA_DEVICE")
            and {"id", "name"} <= columns(db, "TARGET_INFO_GPU")):
        gpu_name = {int(i): name_of(n, ids) for i, n in db.execute("SELECT id, name FROM TARGET_INFO_GPU")}
        for cuda_id, gpu_id in db.execute("SELECT DISTINCT cudaId, gpuId FROM TARGET_INFO_CUDA_DEVICE"):
            if gpu_id is not None and int(gpu_id) in gpu_name:
                names.setdefault(int(cuda_id), gpu_name[int(gpu_id)])

    t0 = min(min(s for s, _ in v) for v in list(kern.values()) + [x for d in copies.values() for x in d.values()] if v)
    t1 = max(max(e for _, e in v) for v in list(kern.values()) + [x for d in copies.values() for x in d.values()] if v)
    span = t1 - t0
    print(f"profile window: {span / 1e9:.3f} s (first to last GPU activity), {len(devices)} card(s)\n")

    merged = {d: merge(kern[d]) for d in devices}
    for d in devices:
        k = merged[d]
        print(f"CUDA device {d}{' (' + names[d] + ')' if d in names else ''}:")
        print(f"  kernels           {total(k) / 1e9:8.3f} s  {100 * total(k) / span:5.1f}% of the window")
        h2d = merge(copies[d].get("RAM->card", []))
        for what in ("RAM->card", "card->RAM", "card->card", "card->card (peer)"):
            iv = merge(copies[d].get(what, []))
            if not iv or copy_bytes[d][what] < 1 << 20:      # nothing, or a few small copies
                continue
            gb = copy_bytes[d][what] / 1e9
            busy = total(iv) / 1e9
            print(f"  {what:<17} {busy:8.3f} s  {100 * busy * 1e9 / span:5.1f}%  {gb:8.2f} GB"
                  f"  {gb / busy if busy else 0:6.2f} GB/s while copying")
        if h2d:
            both = intersect(k, h2d)
            print(f"  RAM->card while kernels ran: {both / 1e9:.3f} s ({100 * both / max(total(h2d), 1):.0f}% of "
                  f"that copying)")
        anything = merge(k + [x for v in copies[d].values() for x in v])
        print(f"  idle (no kernel, no copy) {(span - total(anything)) / 1e9:8.3f} s  "
              f"{100 * (span - total(anything)) / span:5.1f}%")

    # how many cards computed at once
    ev = []
    for d in devices:
        for s, e in merged[d]:
            ev += [(s, 1), (e, -1)]
    ev.sort()
    at = defaultdict(int)
    n = 0
    last = t0
    for t, dlt in ev:
        at[n] += t - last
        n += dlt
        last = t
    at[0] += t1 - last
    print("\ncards computing at the same time:")
    for k in sorted(at):
        print(f"  {k}: {at[k] / 1e9:8.3f} s  {100 * at[k] / span:5.1f}%")

    if has_table(db, "NVTX_EVENTS"):
        cols = columns(db, "NVTX_EVENTS")
        txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in cols else "text"
        rows = db.execute(f"SELECT start, end, {txt} FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
        chunks = [(s, e, t) for s, e, t in rows if t and re.fullmatch(r"chunk \d+\+\d+", str(t))]
        if chunks:
            print("\nprompt chunks (host time from start to every card done):")
            for s, e, t in sorted(chunks):
                pos_t = str(t)[6:]
                tokens = int(pos_t.split("+")[1]) if "+" in pos_t else 0
                secs = (e - s) / 1e9
                print(f"  {t:<24} {secs:8.3f} s" + (f"  {tokens / secs:8.1f} tokens/s" if tokens and secs else ""))
        waits = defaultdict(int)
        for s, e, t in rows:
            if t and not str(t).startswith(("chunk ", "L")):
                waits[str(t)] += e - s
        if waits:
            print("\nhost time in the engine's other ranges:")
            for t, v in sorted(waits.items(), key=lambda x: -x[1]):
                print(f"  {t:<40} {v / 1e9:8.3f} s")

    ph = kernel_phases(db, ids)
    if ph is not None:
        print("\nkernel time by the part of the engine that launched it (seconds; overlapping kernels once):")
        print("  " + " " * 38 + "".join(f"  CUDA {d:<5}" for d in devices))
        for name in PHASES:
            row = [total(merge(ph[d].get(name, []))) / 1e9 for d in devices]
            if any(row):
                print(f"  {name:<38}" + "".join(f"  {v:9.3f}" for v in row))

    # the host's calls into the CUDA runtime (launches, copies, waits): where the host time between kernels goes
    if has_table(db, "CUPTI_ACTIVITY_KIND_RUNTIME") and "nameId" in columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        calls: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for nid, s_, e_ in db.execute("SELECT nameId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            c = calls[re.sub(r"_v\d+$", "", name_of(nid, ids))]
            c[0] += 1
            c[1] += e_ - s_
        if calls:
            print("\nhost calls into the CUDA runtime (all threads):")
            for name, (count, ns) in sorted(calls.items(), key=lambda x: -x[1][1])[:a.top]:
                print(f"  {ns / 1e9:8.3f} s  x{count:<8} {ns / count / 1e3:8.1f} us each  {name}")

    for d in devices:
        print(f"\nCUDA device {d}: the kernels that took the most time")
        tot = total(merged[d]) or 1
        for name, (count, ns) in sorted(kern_by_name[d].items(), key=lambda x: -x[1][1])[:a.top]:
            print(f"  {ns / 1e9:8.3f} s  {100 * ns / tot:5.1f}%  x{count:<7} {name[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
