# Santa 2026 — The Elf Calculator: `scored_0004` in fewer instructions

DAiL bounty `bnty_0730`: beat the house program for Kaggle Santa 2026 puzzle `scored_0004`
(29 targets, value_limit ≈ 2.07e15, 48 registers). Score = emitted instructions, lower is better;
house baseline 313.

## Deliverable

| file | what it is |
|---|---|
| `data/best_scored_0004.csv` | our program — columns `puzzle_id,program`, `program` is a JSON list of `[op, a, b, dst]` instructions (`id,program` copy: `data/best_scored_0004_kaggle.csv`) |
| `data/incumbent_scored_0004.csv` | reference program we started from (400 instructions) |
| `data/puzzles.csv` | the competition puzzle set (public Kaggle data, shipped so verification needs no token) |

## Verify (no API keys, no network)

```bash
python3 test_elf.py                 # rulebook + synthesizer self-checks, incl. replay of the shipped program
python3 -c "import sys; sys.path.insert(0,'src'); import elf_rules as er; \
  p=er.parse_puzzles('src/data/puzzles.csv')['scored_0004']; \
  s=er.parse_submission('src/data/best_scored_0004.csv')['scored_0004']; \
  print(er.check_program(*p[:3], s, p[3]))"
```

`check_program` replays the program instruction by instruction: positive results only, subtraction
only when `a > b`, exact division only, every result ≤ `value_limit`, ≤ 4096 instructions, a target
counts as soon as it appears in any register. It returns `(instruction_count, produced_values)`.

## How the program was built (`elf_solver.py`, stdlib only)

Multiple-constant-multiplication on a register machine: operands shared by several targets are
built once, so the cost is the number of non-start nodes of the union DAG; the DAG is then
pebbled into real register indices (48 registers; a value needed again after being clobbered is
recomputed, which costs extra instructions).

1. **Construction** — cheapest-first plans per target: operands already in the graph, near-start
   base ± small delta, exact factor splits through computed hubs, `h*q + r` linear finishes and a
   binary-ladder fallback; targets processed by descending shared-factor mass.
2. **Recombination** — randomized restarts contribute alternative subtrees into one options pool;
   local search splices other individuals' subtrees and keeps strictly smaller, replay-verified unions.
3. **Exact sharing (CP-SAT)** — recipes harvested from all restarts become a candidate-instruction
   pool; a CP-SAT model picks the minimum instruction set that covers all 29 targets and is closed
   under its own operands (availability + rank-for-acyclicity), with incumbent-improvement cuts.
4. **Width-aware scheduling** — liveness pebbling plus deadline-free list scheduling with register
   reuse; cost model penalises live-register peaks above the cap so narrow DAGs are preferred.
5. Every accepted program is replayed by `elf_rules.check_program` before it is written out.

`run_solver.py` runs the synthesizer over any subset of the puzzle set and writes a
replay-verified submission CSV; `improve.py` re-derives targets against a hub library preloaded
from an existing solution.
