"""test_elf.py — assert-based self-checks for elf_rules + elf_solver (stdlib only).

Run: python test_elf.py   (finishes in ~10-20 s)
"""
from __future__ import annotations

import time

from elf_rules import InvalidProgram, check_program, DEFAULT_REGISTERS
from elf_solver import (_creates_cycle, _is_prime, closure_nodes, factorize, schedule, solve_puzzle,
                        topo_order, to_submission_rows)
import random


def t_rules_ok():
    # 6+35=41, 6*35=210, 210-41=169: targets covered mid-run; final state free
    n, seen = check_program([6, 35], [41, 210, 169], 10**9,
                            [["+", 0, 1, 2], ["*", 0, 1, 3], ["-", 3, 2, 4]])
    assert n == 3 and {41, 210, 169} <= seen


def t_rules_violations():
    bad = [
        ([6, 35], [1], 10**9, [["-", 0, 1, 2]], "not positive"),
        ([6, 35], [1], 10**9, [["/", 1, 0, 2]], "not exact"),
        ([6, 0], [3], 10**9, [["/", 0, 1, 2]], "division by zero"),
        ([6, 35], [7], 10**9, [["+", 0, 1, 2]], "never produced"),
        ([6, 35], [41], 40, [["+", 0, 1, 2]], "value_limit"),
        ([6, 35], [41], 10**9, [["+", 0, 64, 2]], "out of range"),
        ([6, 35], [41], 10**9, [["+", 2, 2, 2]], "never-written"),
        ([6, 35], [41], 10**9, [["^", 0, 1, 2]], "bad operator"),
    ]
    for sv, tg, vl, prog, frag in bad:
        try:
            check_program(sv, tg, vl, prog)
            raise AssertionError(f"missing error for {prog}: {frag}")
        except InvalidProgram as e:
            assert frag in str(e), (frag, str(e))


def t_factorize():
    r = random.Random(0)
    for n in (12, 97, 2**10 * 3**5, 1_000_000_007, 999999999989, 631 * 889 * 1205):
        f = factorize(n, r)
        prod = 1
        for p, e in f.items():
            prod *= p ** e
        assert prod == n and all(_is_prime(p) for p in f), (n, f)



def t_solver_small():
    sv, tg, vl = [6, 35], [41, 210, 9, 10**6 + 3], 10**9
    prog, _cost = solve_puzzle(sv, tg, vl, seed=1, budget_s=3.0, restarts=3)
    assert prog is not None
    inst, _peak = to_submission_rows(prog, sv)
    n, seen = check_program(sv, tg, vl, inst)
    assert set(tg) <= seen and n == len(inst)
    # value-level plan must be topological: operands defined before use
    defined = set(sv)
    for op, a, b, v in prog:
        assert a in defined and b in defined
        defined.add(v)


def t_solver_random_batch():
    rnd = random.Random(42)
    for _ in range(6):
        k = rnd.randint(1, 3)
        sv = sorted(rnd.sample(range(2, 5000), k))
        tg = []
        while len(tg) < 8:
            v = rnd.choice(sv) * rnd.randint(1, 40) + rnd.randint(-60, 60)
            if v > 0:
                tg.append(v)
        vl = max(max(tg), max(sv)) * 4
        prog, _c = solve_puzzle(sv, tg, vl, seed=rnd.randint(1, 99), budget_s=2.0, restarts=2)
        assert prog is not None, (sv, tg)
        inst, _peak = to_submission_rows(prog, sv)
        n, seen = check_program(sv, tg, vl, inst)
        assert set(tg) <= seen


def t_schedule_respects_registers():
    # chain a..: every intermediate stays live until the end (each used twice later) ->
    # with only 3 registers and 2 starts, a wide plan must fail rather than overflow.
    sv = [7, 11]
    prog = [("*", 7, 11, 100), ("+", 100, 7, 101), ("*", 101, 11, 102), ("+", 102, 100, 103),
            ("*", 103, 101, 104)]
    res = schedule(prog, sv, registers=2)
    assert res is None  # 2 starts fill both registers, no free slot
    inst, peak = schedule(prog, sv, registers=48)
    assert peak <= 48 and len(inst) == 5


def t_closure_rejects_broken():
    nodes = {7: None, 11: None, 77: ("*", 7, 11), 99: ("+", 77, 22)}
    try:
        closure_nodes(nodes, [99], {7, 11})
        raise AssertionError("expected KeyError for missing operand")
    except KeyError:
        pass
    sub = closure_nodes(nodes, [77], {7, 11})
    assert set(sub) == {7, 11, 77}
    order = topo_order(sub, [77])
    assert order and order[-1][3] == 77


def t_recombine_cycle_guard():
    # 847's subtree (77*11) depends on 77 -> splicing 847*7 into 77 would loop forever.
    nodes = {7: None, 11: None, 77: ("*", 7, 11), 847: ("*", 77, 11)}
    assert _creates_cycle(nodes, 77, ("*", 847, 7), {7, 11})
    assert not _creates_cycle(nodes, 77, ("+", 7, 11), {7, 11})
    assert not _creates_cycle(nodes, 847, ("+", 77, 11), {7, 11})  # downward use of 77 is fine


def t_ship_program():
    """The shipped data/best_scored_0004.csv must replay under the puzzle's own limits."""
    import os
    from elf_rules import parse_puzzles, parse_submission
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    sv, tg, vl, nreg = parse_puzzles(os.path.join(here, "puzzles.csv"))["scored_0004"]
    prog = parse_submission(os.path.join(here, "best_scored_0004.csv"))["scored_0004"]
    n, seen = check_program(sv, tg, vl, prog, nreg)
    assert set(tg) <= seen and 0 < n <= 4096


if __name__ == "__main__":
    t0 = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print(f"ALL TESTS PASSED in {time.time()-t0:.1f}s (registers={DEFAULT_REGISTERS})")
