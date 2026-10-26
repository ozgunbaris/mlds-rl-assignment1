**T1 transition model.** all 402 rows of P sum to 1: True; from s=200 with 'wait' the whole arrival mass is clipped onto s'=200 (P = 1.00); from s=199 'wait' puts P = 1.00 on the cap (arrivals 1..5 all exceed it).

**T2 agreement.** max|V_enum(t=0) - V_VI| = 3.08e-09, max|V_VI - V_PI| = 1.86e-09, identical policies: True; the policy is a monotone threshold rule (dispatch iff s >= 13): True.

**T3 horizon.** error of the T-period enumeration against V*: T=1: 3.42e+03, T=2: 3.04e+03, T=5: 2.01e+03, T=10: 917, T=20: 244, T=50: 52.2, T=100: 4.02, T=150: 0.309, T=200: 0.0238, T=300: 0.000141, T=500: 3.08e-09. Every measured error lies below the bound gamma^T max V*: True.

**T3b one-period problem.** with T=1 the value is min_a c(s,a) = 2s (never worth paying 100 for at most 30 of holding cost): True.

**T4 threshold enumeration.** evaluating all 202 threshold policies exactly: the minimiser is theta = 13 by V(0) and theta = 13 by the state-average, equal to the PI threshold 13; V(0) = 576.6702 vs PI 576.6702. Dispatching only when full (theta = 15) costs 597.36, never dispatching costs 2203.91.

**T5 Monte Carlo.** s0=0: V=576.67, MC=576.78 +- 0.43 (z=0.49); s0=13: V=676.67, MC=676.54 +- 0.43 (z=-0.57); s0=50: V=1001.53, MC=1001.63 +- 0.43 (z=0.45); s0=100: V=1709.04, MC=1709.29 +- 0.61 (z=0.79); s0=200: V=3821.59, MC=3822.37 +- 1.04 (z=1.47)

**T6 sensitivity.** cf: 25->8, 50->11, 100->13, 200->14, 400->15, 800->201; ch: 0.5->15, 1->14, 2->13, 4->11, 8->8, 16->5; K: 5->5, 10->9, 15->13, 20->16, 30->17, 50->17; gamma: 0.0->201, 0.5->201, 0.8->14, 0.9->13, 0.95->13, 0.99->13

**T7 deterministic arrivals.** A = 3 always: PI threshold 13, cycle of states [3, 6, 9, 12, 15]; V(0) from the linear solve 569.753019 vs geometric-series closed form 569.753019.

**T8 edge cases.** Q(0,dispatch) - Q(0,wait) = 100.000000 (= cf: an empty shuttle costs cf and changes nothing). Raising the cap from 200 to 400 changes V by at most 2.27e-13 for s <= 100 and 4.55e-13 for s <= 200; same policy on s <= 200: True. Partially full dispatches happen at s in [13, 14] (K = 15).

| t | state | action | boarded | left waiting | cost | arrivals | next state |
|---|---|---|---|---|---|---|---|
| 0 | 0 | wait | 0 | 0 | 0 | 3 | 3 |
| 1 | 3 | wait | 0 | 3 | 6 | 5 | 8 |
| 2 | 8 | wait | 0 | 8 | 16 | 1 | 9 |
| 3 | 9 | wait | 0 | 9 | 18 | 5 | 14 |
| 4 | 14 | dispatch | 14 | 0 | 100 | 2 | 2 |
| 5 | 2 | wait | 0 | 2 | 4 | 3 | 5 |
| 6 | 5 | wait | 0 | 5 | 10 | 5 | 10 |
| 7 | 10 | wait | 0 | 10 | 20 | 3 | 13 |
| 8 | 13 | dispatch | 13 | 0 | 100 | 3 | 3 |
| 9 | 3 | wait | 0 | 3 | 6 | 1 | 4 |
| 10 | 4 | wait | 0 | 4 | 8 | 4 | 8 |
| 11 | 8 | wait | 0 | 8 | 16 | 3 | 11 |

**T10 monotone value.** V*(s) is nondecreasing in s: True (increments between -0.000 and 24.852).
