"""Fill the numeric markers of report/pong_sections_draft.md from the Pong result files and
insert the four sections into report/report.md.

    python report/fill_pong.py            # prints the values, reports missing ones, writes report.md
    python report/fill_pong.py --dry      # only prints the values
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from policy_gradient.common import moving_average  # noqa: E402

V = {}
missing = []


def jload(path, default=None):
    if os.path.exists(path):
        return json.load(open(path))
    missing.append(path)
    return default


def rows(path):
    if not os.path.exists(path):
        missing.append(path)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=1):
    return "MISSING" if x is None else f"{x:.{nd}f}"


def lst(xs, nd=1):
    return ", ".join(fmt(x, nd) for x in xs) if xs else "MISSING"


# ---- local runs
summ = jload("results/pong/figures/summary.json", {})
for key, name in [("NONE", "none_lr1e-3"), ("VALUE", "value_lr1e-3")]:
    s = summ.get(name, {})
    V[f"{key}_HOURS"] = fmt(s.get("hours"), 2)
    V[f"{key}_EPISODES"] = f"{s.get('episodes', 0):,}" if s else "MISSING"
    V[f"{key}_STEPS"] = fmt(s.get("env_steps", 0) / 1e6 if s else None, 1)
    V[f"{key}_BEST_MA"] = fmt(s.get("best_ma100"))
    V[f"{key}_FINAL_MA"] = fmt(s.get("final_ma100"))
    ev = s.get("eval_sampled", {})
    V[f"{key}_EVAL_MEAN"] = fmt(ev.get("mean"), 2)
    V[f"{key}_EVAL_STD"] = fmt(ev.get("std"), 2)
    V[f"{key}_EVAL_MIN"] = fmt(ev.get("min"), 0)
    V[f"{key}_EVAL_MAX"] = fmt(ev.get("max"), 0)
    eg = s.get("eval_greedy", {})
    V[f"{key}_GREEDY_MEAN"] = fmt(eg.get("mean"), 2)
    V[f"{key}_GREEDY_STD"] = fmt(eg.get("std"), 2)
    if not ev:
        missing.append(f"eval_sampled for {name}")

# learning-rate comparison of the three short local runs, at the episode count of the shortest
ma = {}
for name in ["none_lr1e-3", "none_lr3e-4", "none_lr1e-4"]:
    r = rows(f"results/pong/{name}/episodes.csv")
    ma[name] = moving_average([float(x["reward"]) for x in r], 100) if r else None
if all(m is not None for m in ma.values()):
    n = min(len(m) for m in ma.values())
    V["LR_EPISODES"] = f"{n:,}"
    V["LR_1E3_MA"] = fmt(ma["none_lr1e-3"][n - 1], 2)
    V["LR_3E4_MA"] = fmt(ma["none_lr3e-4"][n - 1], 2)
    V["LR_1E4_MA"] = fmt(ma["none_lr1e-4"][n - 1], 2)

# training-loop assertion count, action counts, longest game
n_games = 0
maxlen = 0
counts = []
for name in ["none_lr1e-3", "value_lr1e-3", "none_lr3e-4", "value_lr3e-4", "none_lr1e-4"]:
    r = rows(f"results/pong/{name}/episodes.csv")
    n_games += len(r)
    if r:
        maxlen = max(maxlen, max(int(x["length"]) for x in r))
    log = f"results/pong/{name}/train.log"
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("done:") and "counts" in line:
                m = re.search(r"counts \[(\d+), (\d+)\]", line)
                if m:
                    counts = [int(m.group(1)), int(m.group(2))]
V["N_TRAIN_GAMES"] = f"{n_games:,}"
V["N_PONG_EPISODES"] = f"{n_games:,}"
V["N_PONG_MAXLEN"] = f"{maxlen:,}"
V["ACTION_COUNTS"] = f"for the main no-baseline run RIGHT {counts[0]:,} times and LEFT {counts[1]:,} times" if counts else "MISSING"

# ---- validation
val = jload("results/pong/validation/validation.json", {})
pre = val.get("preprocess", {})
V["P1_ACTIVE"] = str(pre.get("active_pixels", "MISSING"))
V["P1_DIFF"] = str(pre.get("diff_nonzero", "MISSING"))
act = val.get("action_semantics", {})
if act:
    V["P2_R0"], V["P2_R1"] = fmt(act["RIGHT_rows"][0]), fmt(act["RIGHT_rows"][1])
    V["P2_L0"], V["P2_L1"] = fmt(act["LEFT_rows"][0]), fmt(act["LEFT_rows"][1])
rp = val.get("random_policy", {})
V["P3_N"] = str(rp.get("episodes", "MISSING"))
V["RANDOM_MEAN"] = fmt(rp.get("mean"), 2)
V["RANDOM_STD"] = fmt(rp.get("std"), 2)
V["RANDOM_MAX"] = fmt(rp.get("max"), 0)
V["RANDOM_LEN"] = fmt(rp.get("mean_length"), 0)
ret = val.get("returns", {})
V["P4_LEN"] = f"{ret.get('length', 0):,}" if ret else "MISSING"
V["P4_GMIN"] = fmt(ret.get("G_min"), 2)
V["P4_GMAX"] = fmt(ret.get("G_max"), 2)
V["P4_G_LOSS"] = fmt(ret.get("G_at_first_loss"), 3)
V["P4_G_LOSS10"] = fmt(ret.get("G_10_steps_before_first_loss"), 3)
if ret.get("G_at_first_loss") and ret.get("G_10_steps_before_first_loss"):
    V["P4_RATIO"] = fmt(ret["G_10_steps_before_first_loss"] / ret["G_at_first_loss"], 3)
pr = val.get("probe", {})
if pr:
    t, r = pr["trained"], pr["random"]
    V["P5_UP_ABOVE"], V["P5_UP_BELOW"] = fmt(t["P_up_given_ball_above"], 3), fmt(t["P_up_given_ball_below"], 3)
    V["P5_DIFF"] = f"{t['P_up_given_ball_above'] - t['P_up_given_ball_below']:+.3f}"
    V["P5_N_ABOVE"], V["P5_N_BELOW"] = f"{t['n_above']:,}", f"{t['n_below']:,}"
    V["P5R_UP_ABOVE"], V["P5R_UP_BELOW"] = fmt(r["P_up_given_ball_above"], 3), fmt(r["P_up_given_ball_below"], 3)
tr = val.get("trace", {})
V["P7_FIRST"] = str(tr["nonzero"][0]) if tr else "MISSING"
gv = val.get("gradient_variance", {})
if gv:
    V["P6_N"] = str(gv["episodes"])
    V["P6_SCORE"] = fmt(gv["mean_score"])
    V["P6_NONE"] = f"{gv['none']['trace_cov']:.3g}"
    V["P6_MEAN"] = f"{gv['mean']['trace_cov']:.3g}"
    V["P6_VALUE"] = f"{gv['value']['trace_cov']:.3g}"
    V["P6_MEAN_X"] = fmt(gv["none"]["trace_cov"] / gv["mean"]["trace_cov"], 1)
    V["P6_VALUE_X"] = fmt(gv["none"]["trace_cov"] / gv["value"]["trace_cov"], 1)
    V["P6_COS_MEAN"] = fmt(gv["mean"]["cos_with_none"], 2)
    V["P6_COS_VALUE"] = fmt(gv["value"]["cos_with_none"], 2)
    V["P6_SNR_NONE"] = fmt(gv["none"]["snr_of_M_episode_average"], 2)
    V["P6_SNR_MEAN"] = fmt(gv["mean"]["snr_of_M_episode_average"], 2)
    V["P6_SNR_VALUE"] = fmt(gv["value"]["snr_of_M_episode_average"], 2)

# ---- cluster seed groups
sg = jload("results/pong/figures/seed_groups.json", {})
for key, label in [("CL_NONE", "no baseline (5 seeds, cluster)"), ("CL_MEAN", "constant baseline (5 seeds, cluster)"), ("CL_VALUE", "value baseline (5 seeds, cluster)")]:
    g = sg.get(label)
    if not g:
        missing.append(f"seed group {label}")
        continue
    eps = g["episodes_per_seed"]
    V[f"{key}_EPISODES_RANGE"] = f"{min(eps):,} to {max(eps):,}"
    V[f"{key}_FINAL_LIST"] = lst(g["final_ma100_per_seed"])
    V[f"{key}_FINAL_MEAN"] = fmt(g["final_ma100_mean"])
    V[f"{key}_FINAL_STD"] = fmt(g["final_ma100_std"])
    ev = g.get("eval", {}).get("sampled")
    V[f"{key}_EVAL_LIST"] = lst(ev["mean_per_seed"]) if ev else "MISSING"
    V[f"{key}_EVAL_MEAN"] = fmt(ev["mean_over_seeds"]) if ev else "MISSING"
lg = jload("results/pong/figures/lr_groups.json", {})
for key, label in [("LRS_1E2", "lr 0.01 (5 seeds)"), ("LRS_1E3", "lr 1e-3 (5 seeds)"), ("LRS_3E4", "lr 3e-4 (5 seeds)")]:
    g = lg.get(label)
    if not g:
        missing.append(f"lr group {label}")
        continue
    V[f"{key}_FINAL_LIST"] = lst(g["final_ma100_per_seed"])
    V[f"{key}_FINAL_MEAN"] = fmt(g["final_ma100_mean"])
    ev = g.get("eval", {}).get("sampled")
    V[f"{key}_EVAL_MEAN"] = fmt(ev["mean_over_seeds"]) if ev else "MISSING"

# advantage-std ratio of the local value run (last 50 updates)
ur = rows("results/pong/value_lr1e-3/updates.csv")
if ur:
    tail = ur[-50:]
    ratio = sum(float(x["adv_std"]) for x in tail) / sum(float(x["G_std"]) for x in tail)
    V["ADV_STD_RATIO"] = fmt(ratio, 2)
    V["ADV_G_STD_LAST"] = fmt(sum(float(x["G_std"]) for x in tail) / len(tail), 2)
    V["ADV_STD_LAST"] = fmt(sum(float(x["adv_std"]) for x in tail) / len(tail), 2)
    V["ENTROPY_VALUE_FINAL"] = fmt(float(ur[-1]["entropy"]), 2)
ur2 = rows("results/pong/none_lr1e-3/updates.csv")
if ur2:
    V["ENTROPY_NONE_FINAL"] = fmt(float(ur2[-1]["entropy"]), 2)

# free-text sentences are filled by hand in report.md after this script runs
for k in ["LOCAL_COMPARISON_SENTENCE", "CLUSTER_COMPARISON_PARAGRAPH", "ADV_RATIO_SENTENCE", "ENTROPY_SENTENCE", "SEED_OUTCOME_SENTENCE", "LR_STUDY_SENTENCE"]:
    V.setdefault(k, f"«{k}»")

draft = open("report/pong_sections_draft.md").read()
needed = sorted(set(re.findall(r"«([A-Z0-9_]*)»", draft)))
unknown = [k for k in needed if k not in V]
for k, v in sorted(V.items()):
    print(f"{k:26s} {v}")
print("\nmissing inputs:", missing)
print("markers without a value:", unknown)
if "--dry" in sys.argv:
    sys.exit(0)
filled = re.sub(r"«([A-Z0-9_]*)»", lambda m: V.get(m.group(1), m.group(0)), draft)
parts = {}
for sec in re.split(r"(?=<!-- PONG_)", filled):
    m = re.match(r"<!-- (PONG_[A-Z0-9]+) -->\n", sec)
    if m:
        parts[m.group(1)] = sec[m.end():].strip()
rep = open("report/report.md").read()
rep = rep.replace("PONG_PART1_PLACEHOLDER", parts["PONG_PART1"]).replace("PONG_TESTS_PLACEHOLDER", parts["PONG_TESTS1"])
rep = rep.replace("PG2_PONG_PLACEHOLDER", parts["PONG_PART2"]).replace("PONG_TESTS2_PLACEHOLDER", parts["PONG_TESTS2"])
rep = rep.replace("N_PONG_EPISODES", V["N_PONG_EPISODES"]).replace("N_PONG_MAXLEN", V["N_PONG_MAXLEN"])
open("report/report.md", "w").write(rep)
print("report.md updated; remaining markers:", sorted(set(re.findall(r"«([A-Z0-9_]*)»", rep))))
