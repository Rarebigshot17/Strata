#!/usr/bin/env python3
"""Conversation parking and checkpoints (src/glm/serve_loop.cpp) against fresh runs, on a small random model.

    tools/glm53_park_check.py ENGINE -m MODEL.gguf [engine options]

ENGINE is strata-glm-ref (CPU) or strata-glm (GPU); MODEL a small GLM-5.3-Flash GGUF (test-llama-archs's, or a
variant from glm53_tiny_variant.py with an indexer top-k small enough to select). Several conversations take turns
on one engine. Every answer must be the tokens a fresh engine (--park-mib 0, reset before each prompt) gives for
the same prompt, and each RESUME must say the state came from where it should have: the live state, a parked
conversation, or a checkpoint.

The GPU engine is bit-reproducible only with --prefill-chunk 0 --no-expert-cache: chunked prompts and the expert
cache change the rounding, so a resumed run and a fresh one may then pick different tokens on a random model.
"""
import random
import re
import subprocess
import sys
import tempfile

N_NEW = 8


class Engine:
    def __init__(self, cmd, log):
        self.cmd = cmd
        self.log = log
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                sys.exit(f"the engine exited before READY: {' '.join(cmd)}")
            if line.startswith("READY"):
                break

    def gen(self, ids, n=N_NEW, **keys):
        """(tokens, resumed, read)."""
        k = "".join(f" {a}={b}" for a, b in keys.items())
        self.p.stdin.write(f"GEN {n}{k} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        toks, resume = [], 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                sys.exit("the engine died")
            f = line.split()
            if f[0] == "T":
                toks.append(int(f[1]))
            elif f[0] == "RESUME":
                resume = int(f[1])
            elif f[0] == "ERR":
                sys.exit(f"engine error: {line.strip()}")
            elif f[0] == "DONE":
                if int(f[8]) != resume:
                    sys.exit(f"DONE says {f[8]} reused, RESUME said {resume}")
                return toks, resume, int(f[14])

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


class Check:
    def __init__(self, engine, ref, sampling):
        self.e, self.ref, self.s = engine, ref, sampling
        self.fail = 0

    def fresh(self, ids):
        self.ref.gen([1 if ids[0] != 1 else 2], n=0, **self.s)     # a different first token: the engine resets
        toks, resume, _ = self.ref.gen(ids, **self.s)
        assert resume == 0
        return toks

    def turn(self, name, ids, expect, **extra):
        toks, resume, read = self.e.gen(ids, **self.s, **extra)
        want = self.fresh(ids)
        ok = toks == want and resume == expect and read == len(ids) - resume
        print(f"{'PASS' if ok else 'FAIL'} {name}: resumed {resume} of {len(ids)} (expected {expect})"
              + ("" if toks == want else f", tokens {toks} != fresh {want}"))
        self.fail |= not ok
        return toks


def rand(rng, n, first=None):
    out = [rng.randrange(3, 120) for _ in range(n)]
    if first is not None:
        out[0] = first
    return out


def scenario(chk, rng):
    """Four conversations taking turns, a regenerated reply, an edited turn, a server-placed checkpoint."""
    A, B, C = rand(rng, 40, 10), rand(rng, 36, 20), rand(rng, 30, 30)
    oA = chk.turn("A: new conversation", A, 0)
    oB = chk.turn("B: new conversation (A parked)", B, 0)
    X = rand(rng, 10)
    A1 = A + oA + X
    oA1 = chk.turn("A: next turn, from A parked", A1, len(A) + N_NEW - 1)
    B1 = B + oB + rand(rng, 6)
    oB1 = chk.turn("B: next turn, from B parked", B1, len(B) + N_NEW - 1)
    again = chk.turn("A: the same turn again, from parked A's checkpoint at its prompt's end", A1, len(A1))
    if again != oA1:
        print("FAIL A: the regenerated reply differs")
        chk.fail = 1
    X2 = rand(rng, 10, X[0] + 1)
    chk.turn("A: the turn edited, from A's checkpoint at its first prompt", A + oA + X2, len(A))
    cut = len(C) - 3
    chk.turn("C: new conversation, checkpoint at cp=", C, 0, cp=cut)
    chk.turn("C: the last 3 prompt tokens changed, from the cp= checkpoint", C[:cut] + rand(rng, 3, C[cut] + 1), cut)
    chk.turn("B: next turn again, from B parked", B1 + oB1 + rand(rng, 5), len(B1) + N_NEW - 1)


def parked_mib(log_path):
    with open(log_path) as f:
        m = re.findall(r"parked \d+ tokens \(([\d.]+) MiB", f.read())
    return float(m[-1]) if m else None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base = sys.argv[1:] + ["--serve", "--ctx", "512"]
    fail = 0
    with tempfile.TemporaryDirectory() as w:
        log = open(f"{w}/engine.log", "w")
        ref = Engine(base + ["--park-mib", "0"], log)
        for label, s in (("greedy", {"temperature": 0}), ("sampled", {"temperature": 0.8, "seed": 7})):
            print(f"== {label}")
            e = Engine(base + ["--checkpoint-min", "8"], log)
            chk = Check(e, ref, s)
            scenario(chk, random.Random(len(label)))
            e.close()
            fail |= chk.fail

        # parking off: the live state is still continued
        print("== --park-mib 0")
        rng = random.Random(3)
        A = rand(rng, 30, 10)
        oA = ref.gen(A)[0]
        toks, resume, _ = ref.gen(A + oA + rand(rng, 4))
        ok = resume == len(A) + N_NEW - 1
        print(f"{'PASS' if ok else 'FAIL'} the live state continued: resumed {resume}")
        fail |= not ok

        # memory: room for two parked conversations (no checkpoints), the least recently used one goes. A parked
        # conversation grows by a few rows per token, so 2.9 of the 37-token ones measured here hold two of the
        # 48-token ones parked later but not three.
        print("== least recently used conversations dropped beyond --park-mib")
        mlog = f"{w}/measure.log"
        with open(mlog, "w") as ml:
            m = Engine(base + ["--checkpoint-min", "100000"], ml)
            rng = random.Random(5)
            m.gen(rand(rng, 30, 10), temperature=0)
            m.gen(rand(rng, 30, 20), temperature=0)
            m.close()
        one = parked_mib(mlog)
        if not one:
            print("FAIL no 'parked' line in the engine log")
            return 1
        e = Engine(base + ["--checkpoint-min", "100000", "--park-mib", f"{2.9 * one:.6f}"], log)
        chk = Check(e, ref, {"temperature": 0})
        rng = random.Random(6)
        P = {k: rand(rng, 30, f) for k, f in (("A", 10), ("B", 20), ("C", 30), ("D", 40))}
        O = {k: chk.turn(f"{k}: new conversation", P[k], 0) for k in "ABCD"}       # A dropped when D came
        cont = {k: P[k] + O[k] + rand(rng, 4) for k in "ABCD"}
        chk.turn("C: from C parked", cont["C"], 30 + N_NEW - 1)                   # parks D; parked: B, D
        chk.turn("A: dropped, read again", cont["A"], 0)                          # parks C, drops B
        chk.turn("D: still parked", cont["D"], 30 + N_NEW - 1)                    # parks A; parked: C, A
        chk.turn("B: dropped, read again", cont["B"], 0)                          # parks D, drops C
        e.close()
        fail |= chk.fail
        ref.close()
        log.close()
        if fail:
            print("--- engine log (last 30 lines)")
            print("".join(open(f"{w}/engine.log").readlines()[-30:]))
    print("all passed" if not fail else "FAILED")
    return fail


if __name__ == "__main__":
    sys.exit(main())
