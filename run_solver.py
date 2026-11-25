"""run_solver.py — solve Santa 2026 puzzles with the elf_solver synthesizer and emit a
replay-verified Kaggle-format submission CSV (id,program).

Usage:
  python run_solver.py --puzzles data/puzzles.csv --out out/submission.csv \
      [--ids scored_0001,scored_0002] [--ids-file ids.txt] [--budget 30] [--seed 1] \
      [--jobs 8] [--seeds 1]

Every emitted program is re-executed by elf_rules.check_program before it is written;
the final CSV is replayed in full and its total printed. Stdlib only, deterministic per seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_rules import InvalidProgram, check_program, parse_puzzles  # noqa: E402
from elf_solver import solve_puzzle, to_submission_rows  # noqa: E402


def _solve_one(job):
    pid, sv, tg, vl, nreg, budget, seed = job
    t0 = time.time()
    best_prog, best_n, peak = None, None, None
    attempt = 0
    b = budget
    while True:  # retry with a longer budget if the cascade could not finish in time
        prog, cost = solve_puzzle(sv, tg, vl, seed=seed + 1000 * attempt, budget_s=b, restarts=6)
        if prog is not None:
            try:
                inst, pk = to_submission_rows(prog, sv)
                n, _ = check_program(sv, tg, vl, inst, nreg)  # hard gate before emission
            except (InvalidProgram, RuntimeError):
                pass  # plan illegal or doesn't fit 48 registers -> next attempt
            else:
                if best_n is None or n < best_n:
                    best_prog, best_n, peak = inst, n, pk
                break
        attempt += 1
        b *= 3
        if attempt > 3:
            break
    return pid, best_prog, best_n, peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="", help="comma-separated puzzle ids (default: all)")
    ap.add_argument("--ids-file", default="", help="file with one id per line")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per puzzle")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1, help="independent runs per puzzle, keep best")
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args(argv)

    puzzles = parse_puzzles(args.puzzles)
    ids = [x for x in args.ids.replace(",", "\n").replace(" ", "\n").splitlines() if x] or sorted(puzzles)
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids = [x.strip() for x in f if x.strip()]
    missing = set(ids) - set(puzzles)
    if missing:
        raise SystemExit(f"unknown ids: {sorted(missing)[:5]}")

    jobs = [(pid, puzzles[pid][0], puzzles[pid][1], puzzles[pid][2], puzzles[pid][3],
             args.budget, args.seed + 1234567 * k)
            for pid in ids for k in range(args.seeds)]

    t0 = time.time()
    best: dict[str, tuple[list, int]] = {}
    with mp.Pool(args.jobs) as pool:
        for i, (pid, prog, n, peak) in enumerate(pool.imap_unordered(_solve_one, jobs), 1):
            if prog is None:
                raise SystemExit(f"solver failed on {pid}")
            if pid not in best or n < best[pid][1]:
                best[pid] = (prog, n)
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time()-t0:.0f}s "
                      f"running-total={sum(v[1] for v in best.values()):,}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in ids:
            w.writerow([pid, json.dumps(best[pid][0], separators=(",", ":"))])

    # full replay of the written CSV — the artifact the house will re-execute
    total = 0
    with open(args.out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sv, tg, vl, nreg = puzzles[row["id"]]
            n, _ = check_program(sv, tg, vl, json.loads(row["program"]), nreg)
            total += n
    print(f"WROTE {args.out}: {len(ids)} puzzles, replay-verified total = {total:,}")
    return total


if __name__ == "__main__":
    main()
