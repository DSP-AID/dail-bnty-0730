"""elf_solver.py — original Santa 2026 Elf Calculator synthesizer (dspaid, DAiL bnty_0713).

Problem = Multiple Constant Multiplication on a register machine: from the starting values build
every target with the fewest +-*/ instructions (exact division only, positive results within the
puzzle's value_limit, 48 registers; a target counts as covered once it appears in any register).

Method — my own synthesis from the puzzle numbers alone (no public program is consumed):

  * Value graph: every constructed value maps to one expression (op, a, b); operands shared by
    several targets are built ONCE, so instructions = non-start nodes of the union DAG.
  * build(v) cascades cheapest-first plans: both operands already in the graph; near-start base
    v = base +- delta (tiny deltas from a ladder seeded by 1 = s//s); exact factor split with one
    computed hub and a recursively built quotient; fresh divisor pair chosen by a cost-to-go
    heuristic; linear finish v = h*q + r over one hub; binary-ladder fallback that always works.
  * Targets processed in descending "shared-factor mass" so common subexpressions appear early.
  * Graph recombination: R randomized constructions per puzzle contribute alternative expressions
    into one options pool; local search swaps a value's expression for another individual's
    subtree and keeps only strictly smaller, replay-verified unions (sub-DAG cross-pollination).
  * Width-aware objective and scheduling: cost = nodes + LAMBDA_WIDTH * max(0, peak_live - CAP);
    bottom-up liveness pebbling (dead-on-arrival targets never hold a register) plus deadline-free
    list schedule reusing slots of dead values allocates real register indices; final programs are
    replayed by elf_rules.check_program before acceptance.

Stdlib only; deterministic for a fixed seed.
"""
from __future__ import annotations

import heapq
import math
import random
import time

LAMBDA_WIDTH = 0.25   # objective penalty per live register above the soft cap (width-aware term)
SOFT_CAP = 40
REGISTERS = 48
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


# --------------------------------------------------------------------- factorization
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factorize(n: int, rnd: random.Random, rho_limit: int = 20000) -> dict[int, int] | None:
    """Full prime factorization; None if the bounded Pollard-rho budget is exhausted."""
    out: dict[int, int] = {}
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        for p in SMALL_PRIMES:
            while m % p == 0:
                out[p] = out.get(p, 0) + 1
                m //= p
        if m == 1:
            continue
        if _is_prime(m):
            out[m] = out.get(m, 0) + 1
            continue
        r = math.isqrt(m)
        if r * r == m and _is_prime(r):
            out[r] = out.get(r, 0) + 2
            continue
        # Brent's Pollard rho with a hard iteration budget
        found = None
        for _try in range(3):
            x = rnd.randrange(2, m - 1)
            y, c = x, rnd.randrange(1, m - 1)

            def f(v: int) -> int:
                return (v * v + c) % m

            d = 1
            for _it in range(rho_limit):
                x, y = f(x), f(f(y))
                if x == y:
                    break
                d = math.gcd(abs(x - y), m)
                if d > 1:
                    break
            if 1 < d < m:
                found = d
                break
        if found is None:
            return None
        stack.append(found)
        stack.append(m // found)
    return out


def all_divisors(fac: dict[int, int], cap: int) -> list[int]:
    ds = [1]
    for p, e in fac.items():
        pe = 1
        new = []
        for _ in range(e):
            pe *= p
            new += [d * pe for d in ds if d <= cap // pe]
        ds += new
    return sorted(d for d in set(ds) if d <= cap)


# --------------------------------------------------------------------- solver core
class Solver:
    def __init__(self, sv: list[int], tg: list[int], vl: int, seed: int, deadline: float,
                 delta_max: int = 60):
        self.sv = sorted(set(sv))
        self.tg = sorted(set(tg))
        self.vl = vl
        self.rnd = random.Random(seed)
        self.deadline = deadline
        self.delta_max = delta_max
        self.node: dict[int, tuple[str, int, int] | None] = {s: None for s in self.sv}
        self.offs = sorted({x for a in self.sv for b in self.sv
                            for x in (a + b, abs(a - b), a * b) if 0 < x <= self.vl})
        self.use_offs = True  # plan_swap flips this to sample both plan families
        self._fac: dict[int, dict[int, int] | None] = {}
        self._div: dict[int, list[int]] = {}

    # -- factorization/divisor cache -----------------------------------------
    def fac(self, n: int) -> dict[int, int] | None:
        if n not in self._fac:
            self._fac[n] = factorize(n, self.rnd) if n > 1 else {}
        return self._fac[n]

    def divs(self, n: int) -> list[int]:
        ds = self._div.get(n)
        if ds is None:
            f = self.fac(n)
            ds = all_divisors(f, min(n, self.vl)) if f is not None else [1]
            self._div[n] = ds
        return ds

    # -- tiny constant ladder (1 = s//s; then doublings/factor splits/additions) --
    def build_tiny(self, k: int, depth: int = 0) -> bool:
        if k in self.node:
            return True
        if time.time() > self.deadline or depth > 16 or len(self.node) > 4200:
            return False
        if k == 1:
            self.node[1] = ("/", self.sv[0], self.sv[0])
            return True
        if k % 2 == 0 and self.build_tiny(k // 2, depth + 1):
            self.node[k] = ("+", k // 2, k // 2)
            return True
        f = self.fac(k)
        if f and sum(f.values()) > 1:
            for d in [d for d in self.divs(k) if 1 < d <= math.isqrt(k)][::-1]:
                if self.build_tiny(d, depth + 1) and self.build_tiny(k // d, depth + 1):
                    self.node[k] = ("*", d, k // d)
                    return True
        parts = sorted((x for x in self.node if 0 < x < k), reverse=True)
        for x in parts[:48]:
            if self.build_tiny(k - x, depth + 1):
                self.node[k] = ("+", x, k - x)
                return True
        for d in range(2, 6):
            if self.build_tiny(d, depth + 2) and abs(k - d) > 1:
                base = k - d
                if base > 0 and self.build_tiny(base, depth + 2):
                    self.node[k] = ("+", base, d)
                    return True
        return False

    # -- cheap cost-to-go heuristic (never does new factorization work) -------
    def est(self, v: int, depth: int = 0) -> float:
        if v in self.node:
            return 0.0
        near = min(abs(v - s) for s in self.sv)
        if 0 < near <= self.delta_max and depth >= 1:
            return 1.0 + min(6.0, 2.0 + 1.44 * math.log2(max(near, 2)))
        ds = self._div.get(v)
        best = 2.0 + 1.44 * math.log2(max(v, 2))  # binary-ladder upper bound
        if ds is not None and depth <= 2:
            for d in ds[1:-1][:40]:
                e_ = v // d
                c = 1.0 + self.est(d, depth + 1) + self.est(e_, depth + 1)
                if c < best:
                    best = c
        return best

    # -- expression cascade ----------------------------------------------------
    def build(self, v: int, depth: int = 0) -> bool:
        if v in self.node:
            return True
        if time.time() > self.deadline or depth > 24 or len(self.node) > 4400 or v > self.vl:
            return False
        if v <= 4 * self.delta_max:
            return self.build_tiny(v, depth)

        comp = self.node
        # (a) both operands already in the graph
        for x in sorted((c for c in comp if c > 1), reverse=True):
            if v % x == 0:
                q = v // x
                if q != x and q in comp:
                    self.node[v] = ("*", x, q)
                    return True
        for x in sorted(comp, reverse=True):
            b = v - x
            if 0 < b and b in comp:
                self.node[v] = ("+", x, b) if x != b else ("+", x, x)
                return True
        for x in comp:
            d_ = x - v
            if 0 < d_ <= 64 and d_ in comp:
                self.node[v] = ("-", x, d_)
                return True

        # (b) near-start base: v = s +- delta with tiny delta
        for s in sorted(self.sv, key=lambda z: abs(z - v)):
            d_ = abs(v - s)
            if 0 < d_ <= self.delta_max and self.build_tiny(d_, depth):
                self.node[v] = ("+", s, d_) if v > s else ("-", s, d_)
                return True

        # (b2) start-pair offset: v -+ w where w is a start-pair combo and the remainder factors
        #      very smooth (probe: every target sits at smooth distance from such a w; deltas here
        #      far exceed delta_max, so no other stage reaches them). Placed ahead of the factor
        #      splits so the offset family wins when eligible; plan_swap re-picks per target.
        #      shortcut: depth-0 only, maxpf<=35k — widen if targets come up empty-handed here.
        if depth == 0 and self.use_offs:
            for w in self.offs:
                for d_ in (v - w, v + w):
                    if not (1 < d_ <= self.vl) or d_ in comp:
                        continue
                    fd = self.fac(d_)
                    if not fd or len(fd) < 2 or max(fd) > 35000:
                        continue
                    if (w in self.node or self.build(w, depth + 1)) and self.build(d_, depth + 1):
                        self.node[v] = ("+", d_, w) if d_ < v else ("-", d_, w)
                        return True

        # (c) exact factor split with one computed hub; build the quotient first
        best_q = None
        for h in (c for c in comp if c > 1):
            if v % h == 0:
                q = v // h
                if q != h and 2 <= q <= self.vl:
                    c_ = 1.0 + self.est(q, depth + 1)
                    if best_q is None or c_ < best_q[0]:
                        best_q = (c_, h, q)
        if best_q and best_q[0] < self.est(v, depth + 2) and self.build(best_q[2], depth + 1):
            self.node[v] = ("*", best_q[1], best_q[2])
            return True

        # (d) fresh divisor pair (both new) via cost-to-go; needs factorization of v
        ds = self.divs(v)
        if len(ds) > 2:
            pairs = sorted(((self.est(d, depth + 1) + self.est(v // d, depth + 1), d)
                            for d in ds[1:-1] if d != v // d), key=lambda p: p[0])
            for _c, lo in pairs[:5]:
                hi = v // lo
                if self.build(lo, depth + 1) and self.build(hi, depth + 1):
                    self.node[v] = ("*", lo, hi)
                    return True

        # (e) Euclidean descent over hubs: v = h*q + r with floor/ceil quotients; the residual r
        #     is recursively buildable (shared descents are the frontier's real currency). Try
        #     hubs largest-first, and score candidates by est(q) + est(r) so cheap descents win.
        cands = []
        for h in sorted((c for c in comp if c > 1), reverse=True)[:200]:
            if h > v:
                continue
            qf = v // h
            for q, r in ((qf, v - qf * h), (qf + 1, v - (qf + 1) * h)):
                if q < 2 or q * h > self.vl:
                    continue
                ar = abs(r)
                if ar == 0:
                    cands.append((self.est(q, depth + 1), h, q, 0))
                elif 0 < ar <= self.vl and (ar in comp or ar <= 4 * self.delta_max or depth < 12):
                    cands.append((self.est(q, depth + 1) + self.est(ar, depth + 1), h, q, r))
        cands.sort()
        for _sc, h, q, r in cands[:8]:
            if not self.build(q, depth + 1):
                continue
            prod = q * h
            if prod not in comp:
                self.node[prod] = ("*", h, q)
            if r == 0:
                if prod != v:
                    self.node[v] = ("*", h, q)
                return True
            ar = abs(r)
            if not (ar in comp or self.build_tiny(ar, depth + 1) or self.build(ar, depth + 2)):
                continue
            if r > 0:
                self.node[v] = ("+", prod, ar)
            else:
                self.node[v] = ("-", prod, ar)
            return True

        # (f) binary ladder fallback: always works once 1 exists
        if not self.build_tiny(1):
            return False
        bits = bin(v)[2:]
        cur = 1
        for b in bits[1:]:
            dbl = cur * 2
            if dbl > self.vl and dbl != v:
                break
            if dbl not in comp:
                self.node[dbl] = ("+", cur, cur)
            cur = dbl
            if b == "1":
                nxt = cur + 1
                if nxt not in comp:
                    if nxt > self.vl:
                        return False
                    self.node[nxt] = ("+", cur, 1)
                cur = nxt
        return v in comp

    # -- full construction pass over all targets -------------------------------
    def construct(self, order: list[int]) -> bool:
        for t in order:
            if not self.build(t):
                return False
        return True


def _creates_cycle(nodes: dict[int, tuple[str, int, int] | None], v: int,
                   e: tuple[str, int, int], sv: set[int]) -> bool:
    """True if assigning nodes[v]=e makes v depend on itself (alt subtree reaches v again)."""
    seen: set[int] = set()
    stack = [e[1], e[2]]
    while stack:
        x = stack.pop()
        if x == v:
            return True
        if x in seen or x in sv:
            continue
        seen.add(x)
        ee = nodes.get(x)
        if ee is not None:
            stack.append(ee[1])
            stack.append(ee[2])
    return False


def closure_nodes(nodes: dict[int, tuple[str, int, int] | None], tg: list[int],
                  sv: set[int]) -> dict[int, tuple[str, int, int] | None]:
    """Minimal sub-DAG producing all targets. Raises KeyError if an operand is unresolvable."""
    out: dict[int, tuple[str, int, int] | None] = {}
    for t in tg:
        if t in out:
            continue
        stack = [t]
        while stack:
            v = stack.pop()
            if v in out:
                continue
            e = nodes.get(v)
            if v in sv and e is None:
                out[v] = None
                continue
            if e is None:
                raise KeyError(v)
            out[v] = e
            stack.append(e[1])
            stack.append(e[2])
    return out


def topo_order(nodes: dict[int, tuple[str, int, int] | None], tg: list[int]):
    order: list[int] = []
    seen: set[int] = set()
    for t in sorted(tg):
        if t in seen or nodes.get(t) is None:
            seen.add(t)
            continue
        stack = [(t, 0)]
        while stack:
            v, stage = stack.pop()
            e = nodes.get(v)
            if e is None or v in seen:
                seen.add(v)
                continue
            if stage == 2:
                seen.add(v)
                order.append(v)
                continue
            stack.append((v, stage + 1))
            for x in (e[1], e[2]):
                if x not in seen:
                    stack.append((x, 0))
    return [(nodes[v][0], nodes[v][1], nodes[v][2], v) for v in order]


def low_pressure_order(prog: list[tuple[str, int, int, int]]):
    """Reorder value-level instructions bottom-up to minimize peak live values (liveness-aware
    pebbling). A value is live across a cut when it is produced below and read above; outputs
    nobody reads (targets covered on the fly) die at once and never hold a register."""
    n = len(prog)
    if not n:
        return []
    tot_reads: dict[int, int] = {}
    prod: dict[int, int] = {}
    for i, (_op, a, b, v) in enumerate(prog):
        tot_reads[a] = tot_reads.get(a, 0) + 1
        tot_reads[b] = tot_reads.get(b, 0) + 1
        prod[v] = i
    unplaced_reads = dict(tot_reads)          # reads still owed by unplaced instructions
    live: set[int] = set()                    # values crossing the current cut
    placed = [False] * n

    def delta(i: int) -> int:
        _op, a, b, v = prog[i]
        return (a not in live) + (b not in live) - (v in live)

    prod_vals = [p[3] for p in prog]
    heap = [(0, -tot_reads.get(v, 0), i) for i, v in enumerate(prod_vals)
            if unplaced_reads.get(v, 0) == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        d, _c, i = heapq.heappop(heap)
        if placed[i]:
            continue
        if (nd := delta(i)) != d:             # stale key — L changed since push; requeue
            heapq.heappush(heap, (nd, _c, i))
            continue
        placed[i] = True
        order.append(i)
        _op, a, b, v = prog[i]
        live.discard(v)
        live.add(a)
        live.add(b)
        for x in (a, b):
            unplaced_reads[x] -= 1
            if unplaced_reads[x] == 0:        # last unplaced reader done -> producer can be placed
                j = prod.get(x)
                if j is not None and not placed[j]:
                    heapq.heappush(heap, (delta(j), -tot_reads.get(x, 0), j))
    if len(order) != n:                       # fallback (should not happen on an SSA DAG)
        return list(prog)
    order.reverse()                           # popped consumers-first -> topological for execution
    return [prog[i] for i in order]


def best_order(nodes: dict[int, tuple[str, int, int] | None], tg: list[int], sv: list[int]):
    """Pick the schedulable ordering with lower width-aware cost: sorted topo or pressure pebble."""
    p1 = topo_order(nodes, tg)
    c1 = width_aware_cost(p1, sv)
    p2 = low_pressure_order(p1)
    if p2 != p1:
        c2 = width_aware_cost(p2, sv)
        if c2 is not None and (c1 is None or c2 < c1):
            return p2, c2
    return p1, c1


# --------------------------------------------------------------------- width-aware scheduler
def schedule(prog: list[tuple[str, int, int, int]], sv: list[int], registers: int = REGISTERS):
    """Demand-driven register allocator with rematerialization: values are materialized on first
    need and evicted (recomputed later from their expression) when slots run out, so ANY DAG
    that fits within the register file is schedulable. Starting values sit at registers
    0..len(sv)-1 exactly as the official machine initializes them. prog must be value-level
    topological; each value appears once. Returns (instructions, peak_live) or None."""
    if not prog:
        return [], len(set(sv))
    # value -> first defining instruction: two nodes can compute the SAME value via different
    # expressions, and a last-wins dict would remat with the other definition's operands.
    expr = {}
    reads: dict[int, list[int]] = {}
    for i, (op, a, b, v) in enumerate(prog):
        expr.setdefault(v, i)
        reads.setdefault(a, []).append(i)
        reads.setdefault(b, []).append(i)
    rpos = {v: 0 for v in reads}

    def next_read(v: int, i: int) -> int:
        """First instruction index >= i where v is still needed (-1 if never again)."""
        lst = reads.get(v, [])
        p = rpos.get(v, len(lst))
        while p < len(lst) and lst[p] < i:
            p += 1
        return lst[p] if p < len(lst) else -1

    start_reg = {v: sv.index(v) for v in set(sv)}
    used = set(start_reg.values())
    if len(used) > registers:
        return None
    resident: dict[int, int] = dict(start_reg)
    free = [r for r in range(registers) if r not in used]
    inst: list[list] = []
    peak = [len(resident)]
    # operand -> #remat frames that will read its register in a not-yet-appended instruction;
    # a pinned slot may not be evicted (stale-register corruption) until every frame emitted.
    pin_cnt: dict[int, int] = {}

    def _pin(v: int) -> None:
        pin_cnt[v] = pin_cnt.get(v, 0) + 1

    def _unpin(v: int) -> None:
        if pin_cnt[v] == 1:
            del pin_cnt[v]
        else:
            pin_cnt[v] -= 1

    def alloc(i: int, protect: set[int]) -> int:
        """A free slot; when full, evict the value needed farthest in the future (dead first)."""
        if free:
            return free.pop()
        cands = [x for x in resident if x not in protect and x not in start_reg and x not in pin_cnt]
        if not cands:
            return -1
        victim = max(cands, key=lambda x: 10 ** 12 if next_read(x, i + 1) == -1 else next_read(x, i + 1))
        return resident.pop(victim)

    def acquire(v: int, i: int, depth: int = 0) -> int:
        """Ensure v sits in a register right before instruction i; rematerialize if evicted."""
        if v in resident:
            return resident[v]
        j = expr.get(v)
        if j is None:
            raise RuntimeError(f"value {v} has no expression and is not a start")
        op, a, b, _pv = prog[j]
        _pin(a)
        _pin(b)             # pin BEFORE recursion: an already-resident operand must also be
        # protected, or a sibling remat's alloc evicts its slot before the pending read (bug seen
        # as 'target never produced' / value_limit blowup in replay)
        sa = acquire(a, i, depth + 1)
        sb = acquire(b, i, depth + 1)
        if sa < 0 or sb < 0:
            _unpin(a)
            _unpin(b)
            return -1
        dst = alloc(i, protect={a, b} | set(resident.keys() & _needed_at(i)))
        if dst < 0:
            _unpin(a)
            _unpin(b)
            return -1
        inst.append([op, sa, sb, dst])
        resident[v] = dst
        peak[0] = max(peak[0], len(resident))
        _unpin(a)
        _unpin(b)
        return dst

    def _needed_at(i: int) -> set[int]:
        """values that instruction i reads (they must not be evicted while being read)."""
        op, a, b, v = prog[i]
        return {a, b}

    for i, (_op, _a, _b, v) in enumerate(prog):
        # release slots of non-starting values dead before this step
        for val in [x for x in list(resident) if next_read(x, i) == -1 and x not in start_reg]:
            free.append(resident.pop(val))
        slot = acquire(v, i)
        if slot < 0:
            return None                        # truly cannot fit
        peak[0] = max(peak[0], len(resident))
        # v itself may already be dead (pure target): its slot stays until next release pass
    return inst, peak[0]


def width_aware_cost(prog: list[tuple[str, int, int, int]], sv: list[int]) -> float | None:
    """instructions + LAMBDA_WIDTH * max(0, peak_live - SOFT_CAP); None if unschedulable."""
    res = schedule(prog, sv)
    if res is None:
        return None
    inst, peak = res
    return len(inst) + LAMBDA_WIDTH * max(0, peak - SOFT_CAP)


# --------------------------------------------------------------------- puzzle driver
def _target_order(tg: list[int], sv: list[int], rnd: random.Random) -> list[int]:
    """Front-load targets that share factors with many others (gcd mass), with jitter."""
    mass = []
    for t in tg:
        m = sum(math.gcd(t, o) > 1 for o in tg if o != t) * 100 \
            + sum(math.gcd(t, s) > 1 for s in sv) * 10 \
            + min(abs(t - s) for s in sv) == 0 and 50
        mass.append((m + rnd.random(), t))
    return [t for _m, t in sorted(mass, reverse=True)]


def solve_puzzle(sv: list[int], tg: list[int], vl: int, seed: int = 1, budget_s: float = 8.0,
                 restarts: int = 3):
    """Returns (value_level_program, cost) — best replay-verifiable program found within budget."""
    deadline = time.time() + budget_s
    sv_u = sorted(set(sv))
    tg_u = sorted(set(tg) - set(sv_u))
    if not tg_u:
        return [], 0.0

    options: dict[int, list[tuple[str, int, int]]] = {}
    best_nodes: dict | None = None
    best_cost = math.inf

    for r in range(max(1, restarts)):
        if time.time() > deadline - budget_s * 0.15:
            break
        rnd = random.Random(seed + 7919 * r)
        delta_max = [60, 24, 120][r % 3]
        s = Solver(sv_u, tg_u, vl, seed + 104729 * r, deadline - budget_s * 0.05, delta_max)
        order = _target_order(tg_u, sv_u, rnd)
        if r:
            rnd.shuffle(order)
        sub_deadline = time.time() + max(0.3, (deadline - time.time()) / max(1, restarts - r))
        s.deadline = min(s.deadline, sub_deadline)
        if not s.construct(order):
            continue
        try:
            nodes = closure_nodes(s.node, tg_u, set(sv_u))
        except KeyError:
            continue
        prog, c = best_order(nodes, tg_u, sv_u)
        if c is None:
            continue
        for v, e in s.node.items():
            if e is not None:
                options.setdefault(v, []).append(e)
        if c < best_cost:
            best_nodes, best_cost = nodes, c

    if best_nodes is None:
        return None, math.inf

    # graph recombination: swap in alternative subtrees from other individuals; accept only
    # strict shrinkage verified by closure + replay-free node count (validity kept by operands
    # always coming from the options pool and transitive closure).
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v, alts in list(options.items()):
            if len(alts) < 2 or v not in best_nodes:
                continue
            for alt in alts:
                if alt == best_nodes.get(v):
                    continue
                if _creates_cycle(best_nodes, v, alt, set(sv_u)):  # cross-individual subtree may loop back into v
                    continue
                trial = dict(best_nodes)
                trial[v] = alt
                try:
                    cand = closure_nodes(trial, tg_u, set(sv_u))
                except KeyError:
                    continue
                prog, c = best_order(cand, tg_u, sv_u)
                if c is not None and c < best_cost - 1e-9:
                    best_nodes, best_cost = cand, c
                    improved = True
                    break
            if improved:
                break

    return best_order(best_nodes, tg_u, sv_u)


def to_submission_rows(prog: list[tuple[str, int, int, int]], sv: list[int]):
    """value-level plan -> register instructions; returns (instructions, peak)."""
    res = schedule(prog, sv)
    if res is None:
        raise RuntimeError("program does not fit in 48 registers")
    return res
