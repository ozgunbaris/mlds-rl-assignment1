"""pytest checks for the shuttle MDP solvers. Run: .venv/bin/python -m pytest tests -q"""
import numpy as np
import pytest

from shuttle_mdp.shuttle import ShuttleMDP, policy_threshold, threshold_policy


@pytest.fixture(scope="module")
def solved():
    mdp = ShuttleMDP()
    V_enum, pi_enum = mdp.enumeration(T=500)
    V_vi, pi_vi, n_vi, res = mdp.value_iteration(tol=1e-10)
    V_pi, pi_pi, n_pi, hist = mdp.policy_iteration()
    return mdp, V_enum, pi_enum, V_vi, pi_vi, V_pi, pi_pi, n_vi, n_pi


def test_transition_matrix_is_stochastic(solved):
    mdp = solved[0]
    assert np.allclose(mdp.P.sum(axis=2), 1.0)
    assert mdp.P[0, mdp.S - 1, mdp.S - 1] == pytest.approx(1.0)  # cap absorbs arrivals


def test_costs(solved):
    mdp = solved[0]
    assert mdp.C[0, 10] == 20 and mdp.C[1, 10] == 100 and mdp.C[1, 20] == 100 + 2 * 5


def test_three_methods_agree(solved):
    mdp, V_enum, pi_enum, V_vi, pi_vi, V_pi, pi_pi, n_vi, n_pi = solved
    assert np.abs(V_enum[0] - V_vi).max() < 1e-6
    assert np.abs(V_vi - V_pi).max() < 1e-6
    assert (pi_enum[0] == pi_vi).all() and (pi_vi == pi_pi).all()


def test_policy_is_monotone_threshold(solved):
    pi_pi = solved[6]
    theta, monotone = policy_threshold(pi_pi)
    assert monotone and 0 < theta < 201


def test_policy_evaluation_matches_value_iteration(solved):
    mdp, _, _, V_vi, _, _, pi_pi, _, _ = solved
    assert np.abs(mdp.policy_evaluation(pi_pi) - V_vi).max() < 1e-6


def test_optimal_threshold_beats_every_other_threshold(solved):
    mdp, _, _, _, _, V_pi, pi_pi, _, _ = solved
    theta, _ = policy_threshold(pi_pi)
    for th in range(0, mdp.S + 1):
        V = mdp.policy_evaluation(threshold_policy(mdp.S, th))
        assert np.all(V >= V_pi - 1e-7)
        if th != theta:
            assert V[0] > V_pi[0]


def test_empty_shuttle_gap_equals_dispatch_cost(solved):
    mdp, _, _, _, _, V_pi, _, _, _ = solved
    Q = mdp.q_values(V_pi)
    assert Q[1, 0] - Q[0, 0] == pytest.approx(mdp.cf)


def test_one_period_problem_is_myopic():
    mdp = ShuttleMDP()
    V1, pi1 = mdp.enumeration(T=1)
    assert np.allclose(V1[0], mdp.C.min(axis=0))
    assert (pi1[0] == 0).all()


def test_gamma_zero_never_dispatches():
    mdp = ShuttleMDP(gamma=0.0)
    _, pol, _, _ = mdp.policy_iteration()
    assert (pol == 0).all()


def test_threshold_moves_with_costs():
    th = lambda **kw: policy_threshold(ShuttleMDP(**kw).policy_iteration()[1])[0]
    assert th(cf=50) <= th(cf=100) <= th(cf=400)
    assert th(ch=8) <= th(ch=2) <= th(ch=0.5)
    assert th(cf=50) < th(cf=400)


def test_deterministic_arrivals_closed_form():
    mdp = ShuttleMDP(arrivals=(3,), arrival_probs=(1.0,))
    V, pol, _, _ = mdp.policy_iteration()
    # follow the deterministic trajectory from 0 until it cycles, sum the series by hand
    s, costs, seen = 0, [], {}
    while s not in seen:
        seen[s] = len(costs)
        a = int(pol[s])
        left = max(s - a * mdp.K, 0)
        costs.append(mdp.cf * a + mdp.ch * left)
        s = min(left + 3, mdp.max_customers)
    start, g = seen[s], mdp.gamma
    pre = sum(c * g ** t for t, c in enumerate(costs[:start]))
    cyc = sum(c * g ** (t - start) for t, c in enumerate(costs[start:], start=start))
    closed = pre + g ** start * cyc / (1 - g ** (len(costs) - start))
    assert V[0] == pytest.approx(closed, rel=1e-9)


def test_monte_carlo_matches_value(solved):
    mdp, _, _, _, _, V_pi, pi_pi, _, _ = solved
    rng = np.random.default_rng(123)
    for s0 in (0, 40, 200):
        x = mdp.simulate(pi_pi, s0, 400, 20000, rng)
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - V_pi[s0]) < 4 * se + 1e-6


def test_cap_does_not_affect_low_states(solved):
    mdp, _, _, _, _, V_pi, pi_pi, _, _ = solved
    big = ShuttleMDP(max_customers=400)
    Vb, pib, _, _ = big.policy_iteration()
    assert np.abs(Vb[:101] - V_pi[:101]).max() < 1e-6
    assert (pib[:201] == pi_pi).all()


def test_value_nondecreasing(solved):
    V_pi = solved[5]
    assert np.all(np.diff(V_pi) >= -1e-9)
