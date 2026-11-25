"""elf_rules.py — Santa 2026 Elf Calculator rulebook (my own implementation).

Machine model: 48 registers holding positive integers; an instruction is
[op, src_a, src_b, dst] with op in + - * / computing x=regs[a], y=regs[b]:
  + -> x+y, - -> x-y, * -> x*y, / -> x//y (only if y != 0 and x % y == 0).
The result must be a positive integer <= the puzzle's value_limit, and is
written to regs[dst] (overwriting). A target is COVERED as soon as it appears
in any register at any point (coverage is order-independent; final state free).

Program constraints: <= 4096 instructions; JSON cell <= 131072 bytes.
Score of a submission = total instruction count over all puzzles (lower better).
"""
from __future__ import annotations

import csv
import json
from typing import Iterable, Sequence

MAX_INSTR = 4096
MAX_CELL_BYTES = 131_072
DEFAULT_REGISTERS = 48


class InvalidProgram(Exception):
    """Raised with the exact rule violated."""


def check_program(starting_values: Sequence[int], targets: Iterable[int], value_limit: int,
                  program: list, registers: int = DEFAULT_REGISTERS) -> tuple[int, set[int]]:
    """Replay one program. Returns (instruction_count, produced_values)."""
    if len(starting_values) > registers:
        raise InvalidProgram(f"{len(starting_values)} starting values exceed {registers} registers")
    if len(program) > MAX_INSTR:
        raise InvalidProgram(f"program has {len(program)} instructions > {MAX_INSTR}")
    cell = json.dumps(program, separators=(",", ":"))
    if len(cell.encode("utf-8")) > MAX_CELL_BYTES:
        raise InvalidProgram(f"program cell is {len(cell)} bytes > {MAX_CELL_BYTES}")

    regs: list[int | None] = [None] * registers
    for i, v in enumerate(starting_values):
        regs[i] = int(v)
    seen: set[int] = set(int(v) for v in starting_values)

    for k, ins in enumerate(program):
        if not isinstance(ins, list) or len(ins) != 4:
            raise InvalidProgram(f"instruction {k}: expected [op,a,b,dst], got {ins!r}")
        op, a, b, d = ins
        if op not in ("+", "-", "*", "/"):
            raise InvalidProgram(f"instruction {k}: bad operator {op!r}")
        for name, idx in (("src_a", a), ("src_b", b), ("dst", d)):
            if not isinstance(idx, int) or not (0 <= idx < registers):
                raise InvalidProgram(f"instruction {k}: {name} register {idx!r} out of range")
        x, y = regs[a], regs[b]
        if x is None:
            raise InvalidProgram(f"instruction {k}: reads never-written register {a}")
        if y is None:
            raise InvalidProgram(f"instruction {k}: reads never-written register {b}")

        if op == "+":
            r = x + y
        elif op == "-":
            r = x - y
        elif op == "*":
            r = x * y
        else:
            if y == 0:
                raise InvalidProgram(f"instruction {k}: division by zero")
            if x % y != 0:
                raise InvalidProgram(f"instruction {k}: {x}/{y} not exact")
            r = x // y

        if r <= 0:
            raise InvalidProgram(f"instruction {k}: result {r} not positive")
        if r > value_limit:
            raise InvalidProgram(f"instruction {k}: result {r} exceeds value_limit {value_limit}")

        regs[d] = r
        seen.add(r)

    missing = set(int(t) for t in targets) - seen
    if missing:
        raise InvalidProgram(f"{len(missing)} target(s) never produced, e.g. {sorted(missing)[:3]}")
    return len(program), seen


def parse_puzzles(path: str) -> dict[str, tuple[list[int], list[int], int, int]]:
    """id -> (starting_values, targets, value_limit, registers)."""
    csv.field_size_limit(10**9)
    out: dict[str, tuple[list[int], list[int], int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = (
                json.loads(row["starting_values"]),
                json.loads(row["targets"]),
                int(row["value_limit"]),
                int(row.get("registers") or DEFAULT_REGISTERS),
            )
    return out


def parse_submission(path: str) -> dict[str, list]:
    """id -> program (list of instructions)."""
    csv.field_size_limit(10**9)
    out: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("puzzle_id") or row["id"]   # the gig asks for puzzle_id, Kaggle uses id
            out[pid] = json.loads(row["program"])
    return out


def replay_submission(submission_path: str, puzzles: dict) -> tuple[int, dict[str, int]]:
    """Replay a whole submission against puzzles. Returns (total, per_puzzle_counts)."""
    subs = parse_submission(submission_path)
    if set(subs) != set(puzzles):
        raise InvalidProgram(f"id mismatch: {len(set(subs) ^ set(puzzles))} differing ids")
    total = 0
    per: dict[str, int] = {}
    for pid, (sv, tg, vl, nreg) in sorted(puzzles.items()):
        n, _ = check_program(sv, tg, vl, subs[pid], nreg)
        per[pid] = n
        total += n
    return total, per
