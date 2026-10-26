**V1 random policy.** 500 episodes with an untrained network: mean 20.3, std 10.4, max 97 (the known level of a uniformly random CartPole policy is about 22).

**V2 controls.** lr = 0 (no learning): first 50 episodes mean 20.2, last 50 mean 18.9; lr = 0.01, normal sign: first 50 episodes mean 25.8, last 50 mean 65.7; lr = 0.01, advantages negated: first 50 episodes mean 12.4, last 50 mean 9.2. With lr = 0 the reward stays at the random level, with negated advantages the agent learns to fail as fast as possible (minimum possible length is about 8-9 steps).

**V3 gradient variance (untrained policy).** trace of the covariance of the per-episode gradient over 200 episodes: no baseline 5212, constant baseline 1954 (2.7x smaller), learned value baseline 1309 (4.0x smaller); cosine between the mean gradients and the no-baseline mean: constant 0.796, value 0.764 (unbiasedness check; 1.0 = same direction); signal-to-noise ratio |mean| / sqrt(trace/M) of the averaged gradient: none 3.03, constant 4.79, value 6.35.

**V3 gradient variance (trained policy).** trace of the covariance of the per-episode gradient over 100 episodes: no baseline 7.927e+05, constant baseline 1.521e+04 (52.1x smaller), learned value baseline 1.52e+04 (52.2x smaller); cosine between the mean gradients and the no-baseline mean: constant 0.946, value 0.934 (unbiasedness check; 1.0 = same direction); signal-to-noise ratio |mean| / sqrt(trace/M) of the averaged gradient: none 1.06, constant 0.67, value 0.80.

**V4 mirror symmetry.** on states visited by the policy, |P(right|s) - P(left|-s)| averages 0.053 (correlation 0.997) for the trained policy and 0.049 (correlation 0.926) for the untrained one. The dynamics are mirror symmetric, so a good policy must be too.

**V4b obvious states.** on 8 states with the pole clearly falling (|theta| >= 0.1 rad and |theta_dot| >= 1 rad/s in the same direction), the trained policy pushes toward the fall with probability 1.000 on average (minimum 1.000); the untrained network: 0.452.

**V5 return scale.** with gamma = 0.95 the discounted return is bounded by 1/(1-gamma) = 20; the largest G_t seen in 20 trained episodes is 20.000. A 20-step survival already gives G_0 = 12.83, a 500-step one gives 20.000: the return only sees roughly the next 20-60 steps, which is why the learning signal is local.

| t | x | x_dot | theta | theta_dot | P(right) | action | reward |
|---|---|---|---|---|---|---|---|
| 0 | -0.020 | +0.037 | -0.049 | +0.032 | 0.48 | right | 1 |
| 1 | -0.019 | +0.233 | -0.049 | -0.276 | 0.07 | left | 1 |
| 2 | -0.015 | +0.039 | -0.054 | +0.001 | 0.39 | left | 1 |
| 3 | -0.014 | -0.156 | -0.054 | +0.276 | 0.88 | left | 1 |
| 4 | -0.017 | -0.350 | -0.049 | +0.551 | 0.99 | right | 1 |
| 5 | -0.024 | -0.154 | -0.038 | +0.244 | 0.87 | right | 1 |
| 6 | -0.027 | +0.042 | -0.033 | -0.061 | 0.31 | left | 1 |
| 7 | -0.026 | -0.153 | -0.034 | +0.221 | 0.84 | right | 1 |
| 8 | -0.029 | +0.043 | -0.030 | -0.082 | 0.28 | left | 1 |
| 9 | -0.028 | -0.152 | -0.031 | +0.201 | 0.82 | right | 1 |

**V6 trace.** first 10 steps of the trained policy: P(right) rises when theta or theta_dot is positive (pole leaning right) and falls otherwise, see the table.
