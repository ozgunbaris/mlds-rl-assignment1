"""pytest checks for the policy-gradient code. Run: .venv/bin/python -m pytest tests -q"""
import ale_py
import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from policy_gradient.cartpole_reinforce import CartPolePolicy, reinforce_loss
from policy_gradient.common import discounted_returns, moving_average
from policy_gradient.pong_reinforce import ACTIONS, ENV_ID, FrameTracker, PongPolicy, make_vec_env, preprocess, select_actions

gym.register_envs(ale_py)


def test_discounted_returns_small_example():
    assert np.allclose(discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0])


def test_discounted_returns_matches_double_loop():
    rng = np.random.default_rng(0)
    r = rng.normal(size=50)
    g = 0.93
    brute = [sum(g ** (k - t) * r[k] for k in range(t, len(r))) for t in range(len(r))]
    assert np.allclose(discounted_returns(r, g), brute)


def test_discounted_returns_reset_on_nonzero():
    r = [0, 0, 1, 0, -1]
    assert np.allclose(discounted_returns(r, 0.9, reset_on_nonzero=True), [0.81, 0.9, 1.0, -0.9, -1.0])
    full = discounted_returns(r, 0.9)
    assert full[2] == pytest.approx(1 - 0.81)


def test_moving_average():
    assert np.allclose(moving_average([1, 2, 3, 4], window=2), [1, 1.5, 2.5, 3.5])
    assert np.allclose(moving_average(np.arange(300), window=100)[-1], np.arange(200, 300).mean())


def test_policy_gradient_step_moves_probabilities_in_the_right_direction():
    torch.manual_seed(0)
    policy = CartPolePolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    s = np.random.default_rng(0).normal(size=(1, 4)).astype(np.float32)
    states = np.concatenate([s, s])
    actions = np.array([0, 1])
    adv = np.array([+1.0, -1.0])  # action 0 was good, action 1 was bad, same state
    with torch.no_grad():
        p_before = F.softmax(policy(torch.as_tensor(s)), 1)[0, 0].item()
    opt.zero_grad()
    reinforce_loss(policy, states, actions, adv).backward()
    opt.step()
    with torch.no_grad():
        p_after = F.softmax(policy(torch.as_tensor(s)), 1)[0, 0].item()
    assert p_after > p_before


def test_zero_advantage_gives_zero_gradient():
    torch.manual_seed(0)
    policy = CartPolePolicy()
    states = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    actions = np.array([0, 1] * 4)
    reinforce_loss(policy, states, actions, np.zeros(8)).backward()
    assert all(p.grad.abs().max().item() == 0.0 for p in policy.parameters())


def test_pong_actions_are_right_and_left():
    env = gym.make(ENV_ID)
    meanings = env.unwrapped.get_action_meanings()
    env.close()
    assert ACTIONS.tolist() == [2, 3]
    assert meanings[2] == "RIGHT" and meanings[3] == "LEFT"


def test_preprocess_shape_values_and_no_side_effect():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(0)
    env.close()
    before = obs.copy()
    x = preprocess(obs)
    assert np.array_equal(obs, before), "preprocess must not modify the frame in place"
    assert x.shape == (80, 80) and x.dtype == np.float32
    assert set(np.unique(x)).issubset({0.0, 1.0})
    assert 0 < x.sum() < 400  # paddles and ball only; background removed


def test_frame_tracker_skips_autoreset_step():
    env = gym.make(ENV_ID)
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(2)
    obs2, *_ = env.step(3)
    env.close()
    tr = FrameTracker(1)
    tr.reset_all(np.stack([obs0]))
    assert np.array_equal(tr.inputs().reshape(80, 80), preprocess(obs0))  # prev is zeros -> input = first frame
    assert tr.observe(0, obs1, done=False) is True
    assert np.array_equal(tr.inputs().reshape(80, 80), preprocess(obs1) - preprocess(obs0))
    assert tr.observe(0, obs2, done=True) is True  # terminal transition is recorded
    assert tr.just_reset[0]
    assert tr.observe(0, obs0, done=False) is False  # autoreset step: not a transition
    assert np.array_equal(tr.inputs().reshape(80, 80), preprocess(obs0))  # diff restarts from zeros


def test_vector_env_episode_bookkeeping_and_pong_reward_structure():
    envs = make_vec_env(2, seed=0, use_async=False)
    obs, _ = envs.reset(seed=0)
    tr = FrameTracker(2)
    tr.reset_all(obs)
    rng = np.random.default_rng(0)
    traj_r = [[], []]
    finished = 0
    while finished < 2:
        a = rng.integers(0, 2, size=2)
        obs, r, term, trunc, info = envs.step(ACTIONS[a])
        for i in range(2):
            if not tr.observe(i, obs[i], bool(term[i] or trunc[i])):
                continue
            traj_r[i].append(float(r[i]))
            if term[i] or trunc[i]:
                rr = np.array(traj_r[i])
                assert info["_episode"][i]
                assert float(info["episode"]["r"][i]) == pytest.approx(rr.sum())
                assert int(info["episode"]["l"][i]) == len(rr)
                wins, losses = int((rr > 0).sum()), int((rr < 0).sum())
                assert set(np.unique(rr)).issubset({-1.0, 0.0, 1.0})
                assert max(wins, losses) == 21 and rr[-1] != 0  # game ends on the 21st point
                assert rr.sum() == wins - losses
                traj_r[i] = []
                finished += 1
    envs.close()


def test_pong_policy_outputs_two_logits_and_samples_valid_indices():
    policy = PongPolicy()
    x = np.zeros((5, 6400), dtype=np.float32)
    a = select_actions(policy, x)
    assert a.shape == (5,) and set(a).issubset({0, 1})
    assert policy(torch.zeros(3, 6400)).shape == (3, 2)
