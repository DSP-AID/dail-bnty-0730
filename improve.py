"""improve.py — hub-pool enrichment for the elf synthesizer.

Takes a value-graph solution, preloads its live values as a hub library, then rebuilds every
target from scratch against that richer graph (targets and anything reading them are dropped
and re-derived). Accepting strict shrinkage of the emitted instruction count — plus lateral
moves at equal cost to diversify the pool — converges past the one-pass construction optimum.

Usage: python improve.py --puzzles data/puzzles.csv --ids scored_0004 --out out/improved.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_rules import InvalidProgram, check_program, parse_puzzles, parse_submission  # noqa: E402
from elf_solver import (  # noqa: E402
    Solver, _target_order, best_order, closure_nodes, to_submission_rows,
)


def nodes_from_program(inst: list[tuple[str, int, int, int]], sv: list[int]) -> dict:
    """Register instructions -> value-level node graph (inverse of schedule's replay)."""
    reg = {r: v for r, v in enumerate(sv)}
    nodes: dict[int, tuple[str, int, int] | None] = {}
    for op, i, j, d in inst:
        a, b = reg[i], reg[j]
        if op == "-":
            a, b = a, b            # keep order: replay already validated a > b
        val = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        nodes[val] = (op, a, b)
        reg[d] = val
    return nodes


def _bootstrap(sv: list[int], tg: list[int], vl: int, seed: int, deadline: float,
               delta_max: int):
    """Plain cascade construction (what solve_puzzle does), but keeping the node graph."""
    rnd = random.Random(seed)
    for _ in range(200):
        if time.time() > deadline - 1.0:
            return None
        s = Solver(sv, tg, vl, rnd.randrange(1 << 30), deadline, delta_max)
        if s.construct(_target_order(tg, sv, rnd)) or s.construct(sorted(tg, reverse=True)):
            try:
                nodes = closure_nodes(s.node, tg, set(sv))
            except KeyError:
                continue
            prog, cost = best_order(nodes, tg, sv)
            if cost is not None:
                return nodes, prog, cost, len(to_submission_rows(prog, sv)[0]), dict(s.node)
    return None


def _rebuild(sv: list[int], tg: list[int], vl: int, pool: dict, order: list[int],
             deadline: float, delta_max: int, seed: int):
    """Re-derive all targets with `pool` (value -> expression) available as free hubs."""
    s = Solver(sv, tg, vl, seed, deadline, delta_max)
    dead = set(tg)
    for v, e in pool.items():
        if v not in dead:
            s.node[v] = e
    for _ in range(20):  # drop derivations that read a dropped value, to fixpoint
        gone = {v for v in dead}
        doomed = [v for v, e in list(s.node.items())
                  if e is not None and (e[1] in gone or e[2] in gone)]
        if not doomed:
            break
        for v in doomed:
            del s.node[v]
        dead |= set(doomed)
    if not s.construct(order):
        return None
    try:
        nodes = closure_nodes(s.node, tg, set(sv))
    except KeyError:
        return None
    prog, cost = best_order(nodes, tg, sv)
    if cost is None:
        return None
    return nodes, prog, cost, len(to_submission_rows(prog, sv)[0]), dict(s.node)


def _rebuild_partial(sv: list[int], tg: list[int], vl: int, t: int, lib: dict,
                     deadline: float, delta_max: int, seed: int):
    """Drop the nodes that ONLY feed target t; re-derive t with everything else — including
    unused library candidates (alternative factorizations) — as free hubs."""
    consumers: dict[int, set[int]] = {}
    for v, e in lib.items():
        if e is None:
            continue
        consumers.setdefault(e[1], set()).add(v)
        consumers.setdefault(e[2], set()).add(v)
    sv_u, tg_u = set(sv), set(tg)
    only_t: set[int] = {t}
    changed = True
    while changed:  # fixpoint: a node dies once every consumer of it is doomed
        changed = False
        for v, e in list(lib.items()):
            if e is None or v in sv_u or v in tg_u or v in only_t:
                continue
            cons = consumers.get(v)
            if cons and cons <= only_t:
                only_t.add(v)
                changed = True
    s = Solver(sv, tg, vl, seed, deadline, delta_max)
    for v, e in lib.items():  # full candidate library; branches under doomed nodes pruned below
        if v not in only_t:
            s.node[v] = e
    for _ in range(30):
        doomed = [v for v, e in list(s.node.items())
                  if e is not None and (e[1] in only_t or e[2] in only_t)]
        if not doomed:
            break
        for v in doomed:
            del s.node[v]
        only_t |= set(doomed)
    if not s.construct([t]):
        return None
    try:
        nodes = closure_nodes(s.node, tg, sv_u)
    except KeyError:
        return None
    prog, cost = best_order(nodes, tg, sv)
    if cost is None:
        return None
    return nodes, prog, cost, len(to_submission_rows(prog, sv)[0]), dict(s.node)


def improve(sv, tg, vl, seed=1, budget_s=120.0, rounds=24, incumbent=None):
    """Returns (nodes, program, emitted). `incumbent` = register instructions to seed the pool with."""
    deadline = time.time() + budget_s
    sv_u = sorted(set(sv))
    tg_u = sorted(set(tg) - set(sv_u))
    if incumbent is not None:
        best_nodes, best_emit = nodes_from_program(incumbent, sv_u), len(incumbent)
        lib = dict(best_nodes)
    else:
        base = _bootstrap(sv_u, tg_u, vl, seed, deadline - 1.0, 60)
        if base is None:
            return None, None, math.inf
        best_nodes, _prog, _cost, best_emit, lib = base
    rnd = random.Random(seed * 31 + 7)
    for _ in range(rounds):
        if time.time() > deadline - 2.0:
            break
        dm = rnd.choice([60, 24, 120])
        if rnd.random() < 0.35:
            res = _rebuild_partial(sv_u, tg_u, vl, rnd.choice(tg_u), lib, deadline - 1.0,
                                   dm, rnd.randrange(1 << 30))
        else:
            order = _target_order(tg_u, sv_u, rnd)
            if rnd.random() < 0.6:
                rnd.shuffle(order)
            pool = best_nodes
            keep = rnd.choice([1.0, 1.0, 0.85, 0.7])
            if keep < 1.0:
                pool = {v: e for v, e in best_nodes.items()
                        if v in set(sv_u) or v in set(tg_u) or rnd.random() < keep}
            res = _rebuild(sv_u, tg_u, vl, pool, order, deadline - 1.0,
                           dm, rnd.randrange(1 << 30))
        if not res:
            continue
        if res[3] < best_emit:
            best_nodes, _prog, _cost, best_emit, lib = res[:5]
        elif res[3] == best_emit and rnd.random() < 0.5:
            # lateral move: equal emitted count but a different graph diversifies the pool,
            # giving later rounds new merge chances (strict-only accept plateaued immediately)
            best_nodes, lib = res[0], res[4]
    prog, _cost = best_order(best_nodes, tg_u, sv_u)
    return best_nodes, prog, best_emit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)  # parallel runs: disjoint seed ranges
    ap.add_argument("--incumbent", help="CSV with a previous program to seed pools from")
    args = ap.parse_args()

    puzzles = parse_puzzles(args.puzzles)
    inc_map = {}
    if args.incumbent:
        inc_map = parse_submission(args.incumbent)
    want = [i.strip() for i in args.ids.split(",") if i.strip()]
    rows, total = [], 0
    for pid in want:
        sv, tg, vl, _nreg = puzzles[pid]
        inc = inc_map.get(pid)
        best = None
        for seed in range(args.seed_offset + 1, args.seed_offset + args.seeds + 1):
            t0 = time.time()
            nodes, prog, cost = improve(sv, tg, vl, seed=seed, budget_s=args.budget,
                                        rounds=args.rounds, incumbent=inc)
            if prog is None:
                print(f"{pid}: seed{seed} FAILED", flush=True)
                continue
            inst, peak = to_submission_rows(prog, sv)
            n, _seen = check_program(sv, tg, vl, inst, _nreg)  # raises InvalidProgram if illegal
            print(f"{pid}: seed{seed} emitted={n} dag={len(prog)} peak={peak} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if best is None or n < best[0]:
                best = (n, inst)
            inc = inst                             # escalate: next seed starts from this
        if best is None:
            print(f"{pid}: FAILED", flush=True)
            continue
        total += best[0]
        prog_json = json.dumps([[op, a, b, d] for op, a, b, d in best[1]])
        cell = f'"{prog_json.replace(chr(34), chr(34) * 2)}"'
        rows.append(f"{pid},{cell}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("id,program\n")
        fh.write("\n".join(rows) + "\n")
    print(f"WROTE {args.out}: total={total}", flush=True)


if __name__ == "__main__":
    main()
