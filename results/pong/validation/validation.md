**P1 preprocessing.** a mid-rally frame keeps 34 of 6400 pixels (two paddles and the ball), values are [0.0, 1.0], and the difference frame has 16 nonzero entries (moving objects only).

**P2 action semantics.** holding RIGHT (2) for 15 steps moves the agent paddle from row 34.5 to 1.0; holding LEFT (3) moves it from row 34.5 to 78.5 (row 0 is the top): RIGHT = up, LEFT = down.

**P3 random policy.** 20 games with uniformly random RIGHT/LEFT: mean score -20.10, std 0.77, best -19, mean length 1105 steps. Every game ended when one side reached 21 and rewards were in {-1, 0, +1}: True.

**P4 discounted return.** over an episode of 5857 steps G_t ranges in [-1.95, 1.55]; at the step of the first lost point G = -0.29919582827939495, ten steps earlier G = -0.27058734401329704 (ratio should be close to 0.99^10 = 0.904 when no other point is close).

**P5 behaviour probe (trained policy).** P(up | ball above paddle) = 0.519 over 12014 frames, P(up | ball below paddle) = 0.296 over 4143 frames; difference +0.223.

**P5 behaviour probe (random policy).** P(up | ball above paddle) = 0.482 over 538 frames, P(up | ball below paddle) = 0.482 over 441 frames; difference -0.001.

**P6 gradient variance.** 12 episodes of the Part-2 policy (mean score 9.3): trace of the per-episode gradient covariance = 0.0008108 with no baseline, 0.0007683 with the constant baseline (1.06x smaller), 0.0002995 with the learned value baseline (2.71x smaller); cosine of the mean gradients with the no-baseline mean: 0.950 and 0.734; signal-to-noise ratio of the 12-episode average gradient: none 1.12, constant 1.21, value 1.27.

| step | nonzero input pixels | P(RIGHT = up) | action | reward |
|---|---|---|---|---|
| 0 | 18 | 0.002 | LEFT (down) | 0 |
| 1 | 18 | 0.072 | LEFT (down) | 0 |
| 2 | 32 | 0.309 | RIGHT (up) | 0 |
| 3 | 20 | 0.985 | RIGHT (up) | 0 |
| 4 | 28 | 0.055 | LEFT (down) | 0 |
| 5 | 4 | 0.395 | LEFT (down) | 0 |
| 6 | 32 | 0.989 | RIGHT (up) | 0 |
| 7 | 4 | 0.826 | RIGHT (up) | 0 |
| 8 | 32 | 0.016 | LEFT (down) | 0 |
| 9 | 4 | 0.395 | RIGHT (up) | 0 |

**P7 trace.** the first input of an episode is the full first frame (both paddles, no previous frame), later inputs only contain the moving pixels; see the table.
