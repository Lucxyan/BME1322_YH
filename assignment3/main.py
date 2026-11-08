"""Reproduce all figures and numerical results in report.md.

Run from the directory containing assignment3/ and the provided glucose_env.py:
    python assignment3/main.py
Only the environment's public reset(), step(), and actions interface is used.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glucose_env import GlucoseEnv  # noqa: E402

from model_based import ModelBasedAgent
from q_learning import QLearningAgent


STATES = [f"B{i}" for i in range(1, 10)]
FIGURES = Path(__file__).resolve().parent / "figures"
DATA_EPISODES = 2000
EVALUATION_EPISODES = 500
ONLINE_EPISODES = 2000
ONLINE_SEEDS = 5
WINDOW = 50
EXPERIMENT_ALPHA = 0.3
EVALUATION_SEED_BASE = 300000


def collect_random_episodes(actions):
    action_rng = random.Random(2026)
    episodes = []
    for index in range(DATA_EPISODES):
        env = GlucoseEnv(seed=2026 + index)
        state = env.reset()
        episode = []
        while True:
            action = action_rng.choice(actions)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        episodes.append(episode)
    return episodes


def evaluate(agent):
    returns = []
    target_steps = 0
    total_steps = 0
    b1_endings = 0
    for index in range(EVALUATION_EPISODES):
        env = GlucoseEnv(seed=EVALUATION_SEED_BASE + index)
        state = env.reset()
        episode_return = 0.0
        while True:
            action = agent.get_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            target_steps += next_state == "B3"
            total_steps += 1
            state = next_state
            if done:
                b1_endings += next_state == "B1"
                break
        returns.append(episode_return)
    return {
        "mean_return": sum(returns) / len(returns),
        "b3_step_fraction": target_steps / total_steps,
        "b1_termination_rate": b1_endings / EVALUATION_EPISODES,
        "mean_steps": total_steps / EVALUATION_EPISODES,
    }


def rolling_mean(values, window):
    running = sum(values[:window])
    result = [running / window]
    for index in range(window, len(values)):
        running += values[index] - values[index - window]
        result.append(running / window)
    return result


def online_trial(actions, seed_index, schedule):
    agent = QLearningAgent(STATES, actions, alpha=EXPERIMENT_ALPHA,
                           seed=8000 + seed_index)
    returns = []
    for episode_index in range(ONLINE_EPISODES):
        env = GlucoseEnv(seed=500000 + seed_index * ONLINE_EPISODES + episode_index)
        state = env.reset()
        total = 0.0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            if done:
                break
        returns.append(total)
        if schedule == "decay":
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    return rolling_mean(returns, WINDOW)


def plot_visits(counts, actions):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.19
    positions = list(range(len(STATES)))
    for action_index, action in enumerate(actions):
        ax.bar([x + (action_index - 1.5) * width for x in positions],
               [counts[f"{state}/{action}"] for state in STATES],
               width=width, label=action)
    ax.set_xticks(positions, STATES)
    ax.set_ylabel("Transition count")
    ax.set_title("Visits per state-action pair, 2,000 random episodes")
    ax.legend(title="Action")
    fig.tight_layout()
    fig.savefig(FIGURES / "visits.png", dpi=180)
    plt.close(fig)


def plot_ablation(ablation, q_metrics, q_default_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = ["500", "1,000", "2,000", "Q a=0.3", "Q a=0.5"]
    values = [ablation[str(n)] for n in (500, 1000, 2000)] + [q_metrics, q_default_metrics]
    fields = ["mean_return", "b3_step_fraction", "b1_termination_rate"]
    titles = ["Mean undiscounted return", "Fraction of steps in B3", "B1 termination rate"]
    for ax, field, title in zip(axes, fields, titles):
        colors = ["#4c78a8"] * 3 + ["#54a24b", "#f58518"]
        ax.bar(labels, [value[field] for value in values], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        if field == "b3_step_fraction":
            ax.set_ylim(0, 1)
        elif field == "b1_termination_rate":
            ax.set_ylim(0, max(0.01, max(value[field] for value in values) * 1.5))
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_and_comparison.png", dpi=180)
    plt.close(fig)


def plot_online(curves):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(WINDOW, ONLINE_EPISODES + 1))
    for schedule, color in (("fixed", "#4c78a8"), ("decay", "#f58518")):
        trials = curves[schedule]
        mean = [sum(trial[i] for trial in trials) / ONLINE_SEEDS for i in range(len(x))]
        sd = [
            (sum((trial[i] - mean[i]) ** 2 for trial in trials) / ONLINE_SEEDS) ** 0.5
            for i in range(len(x))
        ]
        ax.plot(x, mean, color=color, label=schedule)
        ax.fill_between(x, [m - s for m, s in zip(mean, sd)],
                        [m + s for m, s in zip(mean, sd)], color=color, alpha=0.17)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("50-episode rolling mean return")
    ax.set_title("Online Q-learning (alpha=0.3): mean and SD across five seeds")
    ax.legend(title="Epsilon schedule")
    fig.tight_layout()
    fig.savefig(FIGURES / "online_exploration.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    actions = GlucoseEnv(seed=0).actions
    episodes = collect_random_episodes(actions)
    ablation = {}
    model_agents = {}
    for n in (500, 1000, 2000):
        agent = ModelBasedAgent(STATES, actions)
        agent.learn_model(episodes[:n])
        agent.run_value_iteration()
        model_agents[n] = agent
        ablation[str(n)] = evaluate(agent)

    model = model_agents[2000]
    q_agent = QLearningAgent(STATES, actions, alpha=EXPERIMENT_ALPHA)
    q_agent.learn(episodes)
    q_metrics = evaluate(q_agent)
    q_default = QLearningAgent(STATES, actions)
    q_default.learn(episodes)
    q_default_metrics = evaluate(q_default)
    counts = {f"{s}/{a}": model.counts[(s, a)] for s in STATES for a in actions}
    policies = {
        "model": {s: model.get_policy(s) for s in STATES},
        "q_learning": {s: q_agent.get_policy(s) for s in STATES},
    }
    curves = {
        schedule: [online_trial(actions, seed, schedule) for seed in range(ONLINE_SEEDS)]
        for schedule in ("fixed", "decay")
    }
    plot_visits(counts, actions)
    plot_ablation(ablation, q_metrics, q_default_metrics)
    plot_online(curves)
    model_changes = {}
    for n in (500, 1000):
        earlier = model_agents[n]
        keys = [(s, a) for s in STATES for a in actions]
        model_changes[str(n)] = {
            "mean_transition_total_variation_vs_2000": sum(
                0.5 * sum(abs(earlier.get_transition(s, a)[ns] - model.get_transition(s, a)[ns])
                          for ns in STATES)
                for s, a in keys
            ) / len(keys),
            "mean_absolute_reward_difference_vs_2000": sum(
                abs(earlier.get_reward(s, a) - model.get_reward(s, a))
                for s, a in keys
            ) / len(keys),
        }
    output = {
        "offline_q_alpha": EXPERIMENT_ALPHA,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "dataset_transitions": sum(len(episode) for episode in episodes),
        "visit_counts": counts,
        "model_values": {s: model.get_value(s) for s in STATES},
        "model_action_values": {
            s: {a: model._action_value(s, a, model.values) for a in actions}
            for s in STATES
        },
        "q_values": {s: {a: q_agent.get_q_value(s, a) for a in actions} for s in STATES},
        "policies": policies,
        "policy_agreement_nonterminal": sum(
            policies["model"][s] == policies["q_learning"][s] for s in STATES[1:]
        ) / 8,
        "model_ablation": ablation,
        "ablation_policies": {
            str(n): {s: agent.get_policy(s) for s in STATES}
            for n, agent in model_agents.items()
        },
        "model_estimate_change": model_changes,
        "q_evaluation": q_metrics,
        "q_default_sensitivity": {
            "alpha": 0.5,
            "q_values": {s: {a: q_default.get_q_value(s, a) for a in actions} for s in STATES},
            "policy": {s: q_default.get_policy(s) for s in STATES},
            "evaluation": q_default_metrics,
        },
        "online_last_50_rolling_mean_by_seed": {
            schedule: [trial[-1] for trial in trials]
            for schedule, trials in curves.items()
        },
        "online_final_epsilon": {
            "fixed": 0.3,
            "decay": max(0.05, 0.3 * 0.995 ** ONLINE_EPISODES),
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
