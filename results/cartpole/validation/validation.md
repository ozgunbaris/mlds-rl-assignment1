**V1 random policy.** 500 episodes with an untrained network: mean 20.3, std 10.4, max 97 (the known level of a uniformly random CartPole policy is about 22).

**V2 controls.** lr = 0 (no learning): first 50 episodes mean 20.2, last 50 mean 18.9; lr = 0.01, normal sign: first 50 episodes mean 25.8, last 50 mean 65.7; lr = 0.01, advantages negated: first 50 episodes mean 12.4, last 50 mean 9.2. With lr = 0 the reward stays at the random level, with negated advantages the agent learns to fail as fast as possible (minimum possible length is about 8-9 steps).

**V3 gradient variance (untrained policy).** trace of the covariance of the per-episode gradient over 200 episodes: no baseline 5212, constant baseline 1954 (2.7x smaller), learned value baseline 1309 (4.0x smaller); cosine between the mean gradients and the no-baseline mean: constant 0.796, value 0.764 (unbiasedness check; 1.0 = same direction).

**V3 gradient variance (trained policy).** trace of the covariance of the per-episode gradient over 100 episodes: no baseline 8.008e+05, constant baseline 1.215e+04 (65.9x smaller), learned value baseline 9348 (85.7x smaller); cosine between the mean gradients and the no-baseline mean: constant 0.969, value 0.955 (unbiasedness check; 1.0 = same direction).

**V4 mirror symmetry.** on states visited by the policy, |P(right|s) - P(left|-s)| averages 0.491 (correlation 0.099) for the trained policy and 0.049 (correlation 0.926) for the untrained one. The dynamics are mirror symmetric, so a good policy must be too.

**V5 return scale.** with gamma = 0.95 the discounted return is bounded by 1/(1-gamma) = 20; the largest G_t seen in 20 trained episodes is 20.000. A 20-step survival already gives G_0 = 12.83, a 500-step one gives 20.000: the return only sees roughly the next 20-60 steps, which is why the learning signal is local.

| t | x | x_dot | theta | theta_dot | P(right) | action | reward |
|---|---|---|---|---|---|---|---|
| 0 | -0.020 | +0.037 | -0.049 | +0.032 | 0.00 | left | 1 |
| 1 | -0.019 | -0.157 | -0.049 | +0.309 | 0.61 | right | 1 |
| 2 | -0.022 | +0.039 | -0.043 | +0.001 | 0.00 | left | 1 |
| 3 | -0.022 | -0.156 | -0.043 | +0.280 | 0.32 | left | 1 |
| 4 | -0.025 | -0.350 | -0.037 | +0.559 | 1.00 | right | 1 |
| 5 | -0.032 | -0.155 | -0.026 | +0.255 | 0.21 | right | 1 |
| 6 | -0.035 | +0.041 | -0.021 | -0.046 | 0.00 | left | 1 |
| 7 | -0.034 | -0.154 | -0.022 | +0.240 | 0.13 | left | 1 |
| 8 | -0.037 | -0.349 | -0.017 | +0.526 | 1.00 | right | 1 |
| 9 | -0.044 | -0.153 | -0.006 | +0.228 | 0.13 | left | 1 |

**V6 trace.** first 10 steps of the trained policy: P(right) rises when theta or theta_dot is positive (pole leaning right) and falls otherwise, see the table.
