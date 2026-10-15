# Kerf

One-dimensional cutting stock plans against a live remnant inventory.

Kerf takes a cutting list plus a shop's real stock, including every
offcut racked from previous jobs, and returns a cutting plan: which bar
feeds which part, what the new offcuts are, and what the job saved.
Offcuts are not waste in Kerf. They are the next job's stock.

Python 3.10+, numpy and scipy. The integer master problem is solved by
HiGHS through `scipy.optimize.milp`. MIT licensed.

## Status

v0.1.0. 260 tests passing. On the 40 ESICUP Falkenauer instances in
`benchmarks/`, every instance is solved, Kerf uses 6062 bars against
6140 for first-fit decreasing (FFD) and a published lower bound of
6058, and it reaches the published lower bound on 36 of the 40. The
full report is below.

## Install

From a clone:

    pip install .

For the tests and the benchmark driver:

    pip install -e ".[dev]"

## Usage

```python
from datetime import datetime

from kerf.ledger import record_plan_cut
from kerf.models import CuttingList, Material, Origin, Part, StockItem
from kerf.solver import solve

materials = {"al-6061": Material("al-6061", name="6061 bar", unit_cost=0.004)}
inventory = {
    "new-1": StockItem("new-1", "al-6061", 6000),
    "new-2": StockItem("new-2", "al-6061", 6000),
    "rm-17": StockItem("rm-17", "al-6061", 2100, origin=Origin.REMNANT,
                       created_at=datetime(2026, 3, 2)),
}
cutting_list = CuttingList("cl-1042", parts=[
    Part("beam", "al-6061", 2500, 2),
    Part("brace", "al-6061", 950, 6),
    Part("gusset", "al-6061", 400, 4),
])

plan = solve(cutting_list, inventory, materials, kerf_mm=3, min_offcut_mm=50)
for cut in plan.assignments:
    print(cut.stock_item_id, [piece.length for piece in cut.pieces], cut.offcut_length)
```

```
new-1 [2500, 2500, 950] 44
new-2 [950, 950, 950, 400, 400, 400, 400] 1532
rm-17 [950, 950] 197
```

The 2100 mm remnant on the rack carries two braces at no cost; the
rest needs both purchased bars. `plan.metrics` carries utilization, waste
length, new stock used, remnant length consumed, and dollars saved
(here 1900 mm of remnant, $7.60). Every plan passes the verification
oracle in `kerf/verify.py` before `solve` returns it: each cut
reconciles to the millimetre, no bar is cut twice, and every part's
demand is met.

`record_plan_cut(plan, inventory, materials)` then marks the consumed
bars, puts the new offcuts on the rack as remnants that remember their
parent plan, and returns a savings event. A cutting list can also be
read from CSV with `kerf.importer.load_cutting_list`; see
`tests/fixtures/sample_cutting_list.csv` for the format.

When a material cannot be covered by the stock on hand, `solve` raises
`InfeasiblePlanError` naming every such material and the reason, for
example a part longer than every bar of that material.

## How it works

Each material is an independent 1D cutting stock problem. Bars are
grouped into contexts of interchangeable stock: one length, one cost.
Purchased bars cost their length times the material's unit price.
Remnants cost nothing, which is what makes the plan remnant-first: a
free bar that fits is always cheaper than a bought one. Inside a
context, the oldest remnant is cut first.

The solve, in `constraint_engine/geometry/`:

1. Prechecks catch the plainly infeasible cases (no bars, a part
   longer than every bar, more copies than the bars can hold, more
   length than the bars hold) and say which one applies.
2. First-fit decreasing over the contexts, cheapest per millimetre
   first, produces a feasible plan unless it runs out of bars.
3. Column generation over cutting patterns. The master LP chooses how
   many bars of each context to cut with each pattern; pricing is an
   exact bounded knapsack per context. The pool is seeded with
   single-part patterns and the greedy plan's patterns, so the master
   LP is feasible from the first round. The master's duals are highly
   degenerate, so each round prices at several points between the
   current dual and the best-bound dual seen so far (Wentges
   smoothing). Each priced point also gives a Lagrangian and a Farley
   lower bound; once that bound and the LP value round up to the same
   whole number of bars, more columns cannot change the target and the
   loop stops.
4. The integer master over the generated columns, solved with HiGHS.
   If it times out, fails, or finds nothing better than the greedy
   plan, the greedy plan is returned. A solve therefore never returns
   a plan costlier than first-fit decreasing, and whenever the greedy
   plan exists the solve returns a plan.
5. Pattern counts are expanded to physical bars, and the Kerf layer
   turns them into named cuts, remnants, and metrics.

The model is a cover: a plan may cut a few more pieces than ordered
when that costs no extra bar. Overproduced pieces go to the rack as
remnants of their own length.

## Benchmarks

`benchmarks/` holds the 40 integer instances of the ESICUP 1D set as
distributed by Falkenauer (1996): `binpack2.txt` (20 instances of 250
parts) and `binpack3.txt` (20 instances of 500 parts), all on bars of
150. The loader self-checks each file and refuses one that fails. Each
instance is solved twice, through the engine API and through the Kerf
API, with FFD bars plus 5 available, and compared against FFD. Two
seeded scale workloads check time budgets.

Reproduce from the repository root:

    python -m benchmarks.run_benchmarks

The driver exits 0 only when every target holds: aggregate Kerf waste
at most 92% of aggregate FFD waste with every instance resolved, and
both scale workloads within budget.

How to read the waste columns: waste is stock consumed minus the
length of pieces cut. Overproduced pieces count as pieces, since they
go back on the rack, so these columns are lower than the raw leftover.
The bar counts are the stricter comparison. In raw leftover (bars times
150 minus the ordered length) Kerf leaves 4171 mm against FFD's
15871 mm, a 73.7% reduction.

Run on 2026-10-15 on Windows 11, Python 3.12.9, numpy 2.1.3, scipy
1.17.1. The machine was not fully idle (a browser was open, about 20%
CPU load across 16 logical cores), so treat the timings as indicative.
The bar counts and waste figures do not depend on load. Output pasted
unmodified:

```
KERF BENCHMARKS
===============
host     : Windows-11-10.0.26200-SP0
python   : 3.12.9
numpy    : 2.1.3
scipy    : 1.17.1
date     : 2026-10-15

QUALITY: ESICUP Falkenauer 1D vs FFD-decreasing
-----------------------------------------------
instances  : 40 (binpack2.txt + binpack3.txt)
bar budget : FFD bars + 5 fresh bars
settings   : kerf 0 mm, min offcut 1 mm, time limit 5.0 s

label          n   total   LB  FFD  F.waste  Eng  E.waste Kerf  K.waste dBars  dWaste    E.s    K.s
---------------------------------------------------------------------------------------------------
u250_00      250   14783   99  100      217   99       44   99       29    -1    -188   0.89   0.77
u250_01      250   14854  100  101      296  100       52  100       63    -1    -233   0.77   0.71
u250_02      250   15213  102  104      387  102       46  102       66    -2    -321   0.35   0.48
u250_03      250   14914  100  101      236  100       58  100       11    -1    -225   0.91   0.66
u250_04      250   15092  101  102      208  101       58  101       32    -1    -176   0.65   0.57
u250_05      250   15124  101  104      476  102       85  102       11    -2    -465   0.33   0.42
u250_06      250   15154  102  103      296  102       24  102       80    -1    -216   0.30   0.48
u250_07      250   15418  104  105      332  104       53  104       94    -1    -238   0.30   0.30
u250_08      250   15737  105  107      313  106      101  106        8    -1    -305   0.49   0.42
u250_09      250   15030  101  102      270  101       52  101       59    -1    -211   0.75   0.75
u250_10      250   15656  105  106      244  105       94  105       35    -1    -209   0.98   0.92
u250_11      250   15107  101  103      343  101       43  101       43    -2    -300   0.36   0.64
u250_12      250   15739  106  107      311  106       33  106       45    -1    -266   0.48   0.30
u250_13      250   15294  103  104      306  103      117  103       60    -1    -246   0.45   0.47
u250_14      250   14875  100  101      275  100       40  100       42    -1    -233   0.50   0.70
u250_15      250   15722  105  107      328  105       28  105       28    -2    -300   0.40   0.42
u250_16      250   14477   97   99      373   97       73   97       73    -2    -300   0.73   2.39
u250_17      250   14875  100  101      275  100       51  100       44    -1    -231   0.52   0.52
u250_18      250   14955  100  102      345  101       62  100       45    -2    -300   2.51   0.60
u250_19      250   15204  102  103      246  102       32  102       40    -1    -206   0.51   0.40
u500_00      500   29637  198  201      513  198       63  198       63    -3    -450   1.98   1.18
u500_01      500   30127  201  204      473  202       93  202      107    -2    -366   1.83   1.01
u500_02      500   30216  202  205      534  202       84  202       84    -3    -450   1.94   1.61
u500_03      500   30572  204  207      478  204       28  204       28    -3    -450   0.58   0.41
u500_04      500   30767  206  209      583  206       57  206       42    -3    -541   0.90   0.67
u500_05      500   30763  206  207      287  206      110  206       35    -1    -252   0.64   1.25
u500_06      500   31033  207  210      467  207       17  207       17    -3    -450   0.94   0.78
u500_07      500   30597  204  207      453  205       31  205      153    -2    -300   1.36   1.11
u500_08      500   29352  196  199      498  196       48  196       48    -3    -450   1.73   1.77
u500_09      500   30159  202  204      441  202       75  202       69    -2    -372   0.71   1.37
u500_10      500   29860  200  202      440  200       27  200      140    -2    -300   0.69   1.34
u500_11      500   29914  200  203      536  200       66  200       36    -3    -500   1.00   0.73
u500_12      500   29793  199  202      507  199       57  199       57    -3    -450   0.61   2.32
u500_13      500   29338  196  198      362  196       62  196       42    -2    -320   2.60   0.52
u500_14      500   30454  204  206      446  204      123  204       80    -2    -366   1.48   0.80
u500_15      500   30020  201  204      580  201       81  201       71    -3    -509   0.51   2.22
u500_16      500   30151  202  205      599  202       90  202       59    -3    -540   3.65   0.57
u500_17      500   29614  198  201      536  198       22  198       65    -3    -471   0.69   1.65
u500_18      500   30194  202  205      556  202       48  202       64    -3    -492   1.20   0.61
u500_19      500   29345  196  199      505  196       55  196       55    -3    -450   1.55   0.93
---------------------------------------------------------------------------------------------------
TOTAL      15000  905129      6140    15871 6063     2383 6062     2223

engine bars 6063 vs FFD 6140; aggregate relative waste reduction 84.99%
kerf   bars 6062 vs FFD 6140; aggregate relative waste reduction 85.99%
mean per-instance waste gap vs FFD: 1.52 percentage points (40 ok instances)
acceptance: total kerf waste <= 92% of total FFD waste (relative reduction >= 8%) with every instance resolved: MET

SCALE
-----
engine scale : 2000 parts (40 lengths x 50, 20-59 mm), 20 bars of 6000 mm, budget 10.0 s
               solved in 0.01 s, 14 bars used -> PASS
kerf   scale : 200 part lines / 500 items, 20 bars of 6000 mm, budget 3.0 s
               solved in 0.05 s, 15 bars used -> PASS

SCALE RESULT: PASS
```

Before the column generation fix, 11 of the 40 instances (u250_08,
u250_12, u250_13, u250_18, u500_00, u500_03, u500_06, u500_09,
u500_15, u500_16, u500_19) hit the 100-round cap and were reported
unresolved, though FFD packs all of them. The master LP was infeasible
over the single-part seed patterns on the FFD + 5 bar budget, pricing
added one pattern per round at raw degenerate duals, and the solver
gave up when the cap arrived before any feasible LP. Kerf's waste
reduction then was 82.87% over the 29 resolved instances; it is now
85.99% over all 40, with no instance using more bars than FFD.

## Layout

- `constraint_engine/`: the solver library. Core types, 1D geometry,
  bounded knapsack pricing, the greedy plan, master and integer solve,
  the column generation loop, and four public operations (solve,
  explain, replan, diagnose).
- `kerf/`: the shop layer. Domain model, strict CSV cutting list
  importer, solver, verification oracle, remnant ledger, and the
  plan metrics.
- `benchmarks/`: the ESICUP corpus, the FFD baseline, the driver, and
  the two scale workloads.
- `tests/`: unit tests, property tests over seeded random trials, an
  end-to-end round trip through the ledger, and regression tests for
  the eleven instances above.

## Tests

    python -m pytest -q

## License

MIT. See `LICENSE`.
