<!-- PONG_PART1 -->
**Compute.** Two runs were trained on the laptop (Apple M5, 10 cores, CPU only; the 2 runs shared the machine with the CartPole sweeps and the laptop throttled to about 1,300 to 2,000 environment steps per second per run) and fifteen runs on the Deepdish node mlds-deepdish5 through Nomad (36 of its 48 CPU cores reserved, no GPU used, `scripts/nomad/`), with 5 seeds for each of no baseline, constant baseline and value baseline. All runs use the same code and hyperparameters: learning rate $10^{-3}$, batches of 10 games, 4 parallel environments, $\gamma = 0.99$, difference-frame input, 6400-200-2 policy.

**Learning rate.** Three no-baseline runs with learning rates $10^{-3}$, $3 \times 10^{-4}$ and $10^{-4}$ were started together (Figure FIG(compare_episodes)). After about «LR_EPISODES» games the $10^{-3}$ run had a moving average of «LR_1E3_MA» while the two smaller rates were still at «LR_3E4_MA» and «LR_1E4_MA», so the smaller rates were stopped and $10^{-3}$ was kept for every later run.

**Main local run (no baseline).** The run trained for «NONE_HOURS» hours, «NONE_EPISODES» games and «NONE_STEPS» million environment steps. The 100-game moving average rose from $-20.4$ to a best value of «NONE_BEST_MA» and ended at «NONE_FINAL_MA» (Figure FIG(none_lr1e-3_training)). The 500-game roll-out of the final network scores «NONE_EVAL_MEAN» on average with standard deviation «NONE_EVAL_STD» (minimum «NONE_EVAL_MIN», maximum «NONE_EVAL_MAX»; Figure FIG(none_lr1e-3_hist_sampled)); with greedy actions the mean is «NONE_GREEDY_MEAN» and the standard deviation «NONE_GREEDY_STD». A uniformly random RIGHT/LEFT policy scores «RANDOM_MEAN» (standard deviation «RANDOM_STD»; test P3).

![Part 1, Pong-v5, no baseline, lr 1e-3 (local run): episode reward and its 100-episode moving average.](../results/pong/figures/none_lr1e-3_training.png){ width=80% }

![Part 1, Pong-v5, no baseline (local run): 500 roll-out games of the final policy with sampled actions.](../results/pong/figures/none_lr1e-3_hist_sampled.png){ width=62% }

![Pong-v5: 100-episode moving-average reward of all local runs, including the three learning rates compared at the start.](../results/pong/figures/compare_episodes.png){ width=80% }

**Five seeds on the cluster (no baseline).** The five cluster runs trained for 4.0 hours each («CL_NONE_EPISODES_RANGE» games). Their final moving averages are «CL_NONE_FINAL_LIST» (mean «CL_NONE_FINAL_MEAN», standard deviation over seeds «CL_NONE_FINAL_STD») and their 500-game roll-outs «CL_NONE_EVAL_LIST» (mean over seeds «CL_NONE_EVAL_MEAN»). Figure FIG(compare_seed_groups) shows the mean and the spread over seeds next to the two baseline variants of Part 2.

<!-- PONG_TESTS1 -->
1. **Preprocessing (P1, Figure FIG(fig_val_pong_preprocess)).** A mid-rally frame keeps «P1_ACTIVE» of the 6400 pixels (two paddles and the ball), all values are 0 or 1, and the difference frame has «P1_DIFF» nonzero entries: the moving objects only. The unit tests check the shape, the binary values and that the input frame is left unchanged.
2. **Action semantics (P2).** Holding action 2 (RIGHT) for 15 steps moves the agent's paddle from row «P2_R0» to row «P2_R1» of the 80 x 80 image; holding action 3 (LEFT) moves it from «P2_L0» to «P2_L1». Row 0 is the top, so RIGHT is up and LEFT is down; the behaviour probe below relies on this.
3. **Random policy and game structure (P3).** «P3_N» games with uniformly random RIGHT/LEFT score «RANDOM_MEAN» on average (standard deviation «RANDOM_STD», best «RANDOM_MAX», mean length «RANDOM_LEN» steps). Every game ended when one side reached 21 points, the last reward was nonzero and all rewards were in $\{-1, 0, 1\}$; the same structural assertions run on every game inside the training loop, and the loop also asserts that its own episode reward and length equal gymnasium's `RecordEpisodeStatistics` values. These assertions held for all «N_TRAIN_GAMES» training games of the local runs. The action counters at the end of the runs show only actions 2 and 3 («ACTION_COUNTS»).
4. **Discounted return along a game (P4, Figure FIG(fig_val_pong_returns)).** Over a «P4_LEN»-step game of the trained policy $G_t$ ranges in [«P4_GMIN», «P4_GMAX»]; at the first lost point $G_t$ = «P4_G_LOSS» and ten steps earlier «P4_G_LOSS10», a ratio of «P4_RATIO» against $0.99^{10} = 0.904$ expected when no other point is near. The sawtooth shape shows that each return is dominated by the next point even though it is computed over the whole game.
5. **Behaviour probe (P5).** On frames where both the ball and the agent's paddle are visible, the trained policy chooses "up" with probability «P5_UP_ABOVE» when the ball is above the paddle and «P5_UP_BELOW» when it is below (difference «P5_DIFF», «P5_N_ABOVE» and «P5_N_BELOW» frames); the untrained network gives «P5R_UP_ABOVE» and «P5R_UP_BELOW». The agent moves the paddle toward the ball, which is the behaviour the score requires and which the pixels alone do not reveal.
6. **First steps of a game (P7).** The first input of a game holds the full first frame («P7_FIRST» nonzero pixels, both paddles) and later inputs only the moving pixels (3 to 20 nonzero entries); the table in `results/pong/validation/validation.md` lists the first 10 inputs, action probabilities and actions.
7. **Longest game and truncation.** The longest recorded game had «N_PONG_MAXLEN» steps, far below the 27,000-step frame limit, so no game was truncated.

![Test P1: raw frame, preprocessed 80 x 80 frame, and the difference frame fed to the network.](../results/pong/validation/fig_val_pong_preprocess.png){ width=90% }

![Test P4: discounted return along one trained-policy game, with the won and lost points marked.](../results/pong/validation/fig_val_pong_returns.png){ width=80% }

<!-- PONG_PART2 -->
**Main local run (value baseline).** With the same learning rate and batch size the value-baseline run trained for «VALUE_HOURS» hours, «VALUE_EPISODES» games and «VALUE_STEPS» million steps; its moving average reached a best of «VALUE_BEST_MA» and ended at «VALUE_FINAL_MA» (Figure FIG(value_lr1e-3_training)). The 500-game roll-out scores «VALUE_EVAL_MEAN» on average with standard deviation «VALUE_EVAL_STD» (minimum «VALUE_EVAL_MIN», maximum «VALUE_EVAL_MAX»; Figure FIG(value_lr1e-3_hist_sampled)); with greedy actions «VALUE_GREEDY_MEAN» and «VALUE_GREEDY_STD». Figure FIG(compare_env_steps) compares the two local runs on the same number of environment steps and Figure FIG(compare_wall_time) on wall-clock time: «LOCAL_COMPARISON_SENTENCE»

![Part 2, Pong-v5, value baseline, lr 1e-3 (local run): episode reward and its 100-episode moving average.](../results/pong/figures/value_lr1e-3_training.png){ width=80% }

![Part 2, Pong-v5, value baseline (local run): 500 roll-out games of the final policy with sampled actions.](../results/pong/figures/value_lr1e-3_hist_sampled.png){ width=62% }

![Pong-v5: moving-average reward of the two local runs against environment steps.](../results/pong/figures/compare_env_steps.png){ width=78% }

![Pong-v5: moving-average reward of the two local runs against training wall-clock time.](../results/pong/figures/compare_wall_time.png){ width=78% }

**Five seeds per variant on the cluster.** Figure FIG(compare_seed_groups) and the table give the cluster results after 4.0 hours of training per run.

| variant (5 seeds each) | games per seed | final MA100 per seed | final MA100 mean (std over seeds) | roll-out mean per seed (500 games, sampled actions) | roll-out mean over seeds |
|---|---|---|---|---|---|
| no baseline | «CL_NONE_EPISODES_RANGE» | «CL_NONE_FINAL_LIST» | «CL_NONE_FINAL_MEAN» («CL_NONE_FINAL_STD») | «CL_NONE_EVAL_LIST» | «CL_NONE_EVAL_MEAN» |
| constant baseline | «CL_MEAN_EPISODES_RANGE» | «CL_MEAN_FINAL_LIST» | «CL_MEAN_FINAL_MEAN» («CL_MEAN_FINAL_STD») | «CL_MEAN_EVAL_LIST» | «CL_MEAN_EVAL_MEAN» |
| value baseline | «CL_VALUE_EPISODES_RANGE» | «CL_VALUE_FINAL_LIST» | «CL_VALUE_FINAL_MEAN» («CL_VALUE_FINAL_STD») | «CL_VALUE_EVAL_LIST» | «CL_VALUE_EVAL_MEAN» |

«CLUSTER_COMPARISON_PARAGRAPH»

![Pong-v5, cluster: mean and standard deviation over 5 seeds of the 100-episode moving-average reward for no baseline, the constant baseline and the learned value baseline.](../results/pong/figures/compare_seed_groups.png){ width=82% }

<!-- PONG_TESTS2 -->
1. **Spread of the advantage during training (Figure FIG(compare_advantage_std)).** Every update logs the standard deviation of $G_t$ and of the advantage over the batch. Without a baseline the two coincide by definition. With the value baseline the advantage spread is «ADV_RATIO_SENTENCE»
2. **Variance of the gradient estimate (P6).** For «P6_N» games of the Part 2 policy (mean score «P6_SCORE») the trace of the per-game gradient covariance is «P6_NONE» with no baseline, «P6_MEAN» with the constant baseline («P6_MEAN_X»x smaller) and «P6_VALUE» with the learned value baseline («P6_VALUE_X»x smaller); the cosines between the mean gradients and the no-baseline mean are «P6_COS_MEAN» and «P6_COS_VALUE», and the signal-to-noise ratio of the «P6_N»-game average gradient is «P6_SNR_NONE», «P6_SNR_MEAN» and «P6_SNR_VALUE».
3. **Policy entropy (Figure FIG(compare_entropy)).** «ENTROPY_SENTENCE»
4. **Outcome across seeds.** «SEED_OUTCOME_SENTENCE»

![Pong-v5: standard deviation over each batch of the return G_t and of the advantage actually used, for the two local runs.](../results/pong/figures/compare_advantage_std.png){ width=80% }

![Pong-v5: mean policy entropy per update for the two local runs.](../results/pong/figures/compare_entropy.png){ width=78% }
