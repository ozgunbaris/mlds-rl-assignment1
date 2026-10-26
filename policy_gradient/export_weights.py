"""Save only the policy (and value) state dicts of a training checkpoint, without optimizer state.

    python -m policy_gradient.export_weights results/pong/none_lr1e-3/checkpoint.pt results/pong/none_lr1e-3/policy_weights.pt
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
out = {"policy": ck["policy"], "value": ck.get("value"), "episodes": ck["state"]["episodes"], "updates": ck["state"]["updates"]}
torch.save(out, dst)
print(f"saved {dst}: episodes={out['episodes']} updates={out['updates']}")
