"""Shuttle dispatch problem as a finite Markov decision process.

State  s in {0, ..., max_customers}: customers waiting at the start of a period.
Action a in {0, 1}: 1 = dispatch a shuttle of capacity K, 0 = wait.
Cost   c(s, a) = cf * a + ch * max(s - a*K, 0)   (holding cost charged after dispatch)
Next   s' = min(max(s - a*K, 0) + A, max_customers),  A ~ Uniform{1,...,5}

Everything is a cost, so the solvers *minimise* expected discounted cost.
"""
import numpy as np


class ShuttleMDP:
    def __init__(self, K=15, cf=100.0, ch=2.0, arrivals=(1, 2, 3, 4, 5), arrival_probs=None,
                 max_customers=200, gamma=0.95):
        self.K = int(K)
        self.cf = float(cf)
        self.ch = float(ch)
        self.max_customers = int(max_customers)
        self.gamma = float(gamma)
        self.arrivals = np.asarray(arrivals, dtype=int)
        if arrival_probs is None:
            arrival_probs = np.full(len(self.arrivals), 1.0 / len(self.arrivals))
        self.arrival_probs = np.asarray(arrival_probs, dtype=float)
        assert abs(self.arrival_probs.sum() - 1.0) < 1e-12
        self.S = self.max_customers + 1
        self.n_actions = 2
        self.P = np.zeros((2, self.S, self.S))  # P[a, s, s']
        self.C = np.zeros((2, self.S))  # C[a, s]
        for s in range(self.S):
            for a in (0, 1):
                left = max(s - a * self.K, 0)
                self.C[a, s] = self.cf * a + self.ch * left
                for arr, p in zip(self.arrivals, self.arrival_probs):
                    s_next = min(left + arr, self.max_customers)
                    self.P[a, s, s_next] += p

    # ------------------------------------------------------------------ helpers
    def q_values(self, V):
        """Q[a, s] = c(s,a) + gamma * E[V(s') | s, a]."""
        return self.C + self.gamma * (self.P @ V)

    def greedy(self, V):
        return self.q_values(V).argmin(axis=0)

    # ------------------------------------------------------- a) enumeration
    def enumeration(self, T=500, terminal_value=None):
        """Finite-horizon backward induction over T periods (exhaustive enumeration of
        every state-action pair at every stage). Returns V with shape (T+1, S) and the
        stage policies with shape (T, S). V[0] is the value function at time 0."""
        V = np.zeros((T + 1, self.S))
        if terminal_value is not None:
            V[T] = terminal_value
        policy = np.zeros((T, self.S), dtype=int)
        for t in range(T - 1, -1, -1):
            Q = self.q_values(V[t + 1])
            V[t] = Q.min(axis=0)
            policy[t] = Q.argmin(axis=0)
        return V, policy

    # --------------------------------------------------- b) value iteration
    def value_iteration(self, tol=1e-10, max_iter=100000, V0=None):
        """Successive approximation V <- min_a [c + gamma P V]. Stops when the sup-norm
        change is below tol. Returns V, greedy policy, iterations, residual history."""
        V = np.zeros(self.S) if V0 is None else np.array(V0, dtype=float)
        residuals = []
        for it in range(1, max_iter + 1):
            V_new = self.q_values(V).min(axis=0)
            res = float(np.max(np.abs(V_new - V)))
            residuals.append(res)
            V = V_new
            if res < tol:
                break
        return V, self.greedy(V), it, np.array(residuals)

    # -------------------------------------------------- c) policy iteration
    def policy_evaluation(self, policy):
        """Exact evaluation: solve (I - gamma P_pi) V = c_pi."""
        idx = np.arange(self.S)
        P_pi = self.P[policy, idx]
        C_pi = self.C[policy, idx]
        return np.linalg.solve(np.eye(self.S) - self.gamma * P_pi, C_pi)

    def policy_iteration(self, policy0=None, max_iter=1000):
        """Howard's policy iteration with exact evaluation. The action is only changed
        when strictly better (tie -> keep), which rules out cycling. Returns V, policy,
        iterations, and the list of policies visited (first = initial)."""
        policy = np.zeros(self.S, dtype=int) if policy0 is None else np.array(policy0, dtype=int)
        history = [policy.copy()]
        for it in range(1, max_iter + 1):
            V = self.policy_evaluation(policy)
            Q = self.q_values(V)
            idx = np.arange(self.S)
            q_current = Q[policy, idx]
            best = Q.argmin(axis=0)
            improve = Q[best, idx] < q_current - 1e-9
            new_policy = np.where(improve, best, policy)
            history.append(new_policy.copy())
            if not improve.any():
                return V, policy, it, history
            policy = new_policy
        raise RuntimeError("policy iteration did not converge")

    # ------------------------------------------------------------- simulation
    def simulate(self, policy, s0, T, n_episodes, rng):
        """Monte Carlo discounted cost of a stationary policy over T periods."""
        s = np.full(n_episodes, s0, dtype=int)
        total = np.zeros(n_episodes)
        disc = 1.0
        for t in range(T):
            a = policy[s]
            left = np.maximum(s - a * self.K, 0)
            total += disc * (self.cf * a + self.ch * left)
            arr = rng.choice(self.arrivals, size=n_episodes, p=self.arrival_probs)
            s = np.minimum(left + arr, self.max_customers)
            disc *= self.gamma
        return total

    def trace(self, policy, s0, T, rng):
        """Step-by-step trajectory (state, action, cost, arrivals, next state)."""
        rows = []
        s = int(s0)
        for t in range(T):
            a = int(policy[s])
            left = max(s - a * self.K, 0)
            cost = self.cf * a + self.ch * left
            arr = int(rng.choice(self.arrivals, p=self.arrival_probs))
            s_next = min(left + arr, self.max_customers)
            rows.append({"t": t, "state": s, "action": a, "boarded": s - left, "left_waiting": left,
                         "cost": cost, "arrivals": arr, "next_state": s_next})
            s = s_next
        return rows


def threshold_policy(S, theta):
    """Dispatch iff s >= theta (theta = S means never dispatch)."""
    return (np.arange(S) >= theta).astype(int)


def policy_threshold(policy):
    """Smallest state at which the policy dispatches; S if it never dispatches.
    Also returns whether the policy is a pure threshold (monotone) policy."""
    S = len(policy)
    dispatch = np.flatnonzero(policy == 1)
    if len(dispatch) == 0:
        return S, True
    theta = int(dispatch[0])
    monotone = bool(np.all(policy[theta:] == 1))
    return theta, monotone
