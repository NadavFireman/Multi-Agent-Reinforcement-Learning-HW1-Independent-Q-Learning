import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from iql import IQL
from train_iql import iql_eval
from utils import (ACTION_LABELS, OUTPUT_DIR, SCHEDULE_CSV_NAME,
                   FIG_WIDTH, FIG_HEIGHT, FIG_WSPACE)


SCHEDULES = ["linear", "exponential", "constant"]
SCHEDULE_LABELS = {"linear": "Linear decay",
                   "exponential": "Exponential decay",
                   "constant": "Constant (0.1)"}


class ScheduledIQL(IQL):
    def __init__(self, *args, schedule="linear", **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        if self.schedule == "linear":
            self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
        elif self.schedule == "exponential":
            self.epsilon = max(0.01, 0.01 ** (timestep / (0.8 * max_timestep)))
        elif self.schedule == "constant":
            self.epsilon = 0.1


def train_schedule(env, config, schedule):
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    agents = ScheduledIQL(num_agents=env.n_agents,
                          action_spaces=env.action_space,
                          gamma=config["gamma"],
                          learning_rate=config["lr"],
                          epsilon=config["init_epsilon"],
                          schedule=schedule)

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    epsilon_trace = []
    eval_means = []
    eval_q_tables = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            if step_counter % config["ep_length"] == 0:
                epsilon_trace.append(agents.epsilon)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            obss = n_obss

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, _ = iql_eval(env, config, agents.q_tables, output=False)
            eval_means.append(mean_return)
            eval_q_tables.append(copy.deepcopy(agents.q_tables))

    return {"epsilon_trace": epsilon_trace,
            "eval_means": eval_means,
            "eval_q_tables": eval_q_tables,
            "q_tables": agents.q_tables}


def save_schedule_csv(schedule_results, config, path=None, output=True):
    """
    Write the exploration schedule comparison to a CSV file

    One row per schedule and evaluation point, holding the epsilon of the
    schedule, the mean evaluation returns and the Q-values of both agents.

    :param schedule_results (Dict[str, Dict]): output of train_schedule for each schedule
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param path (str): path of the CSV file to write
    :param output (bool): flag whether the written path should be printed
    :return (str): path of the written CSV file
    """
    if path is None:
        path = os.path.join(OUTPUT_DIR, SCHEDULE_CSV_NAME)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    n_agents = len(next(iter(schedule_results.values()))["eval_means"][0])
    n_actions = len(ACTION_LABELS)

    header = ["schedule", "evaluation", "episode", "epsilon"]
    header += [f"agent{i + 1}_return_mean" for i in range(n_agents)]
    for i in range(n_agents):
        header += [f"agent{i + 1}_q_{ACTION_LABELS[a].lower()}" for a in range(n_actions)]

    n_rows = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for schedule in SCHEDULES:
            results = schedule_results[schedule]
            epsilon_trace = results["epsilon_trace"]

            for e, (mean, q_tables) in enumerate(zip(results["eval_means"],
                                                     results["eval_q_tables"])):
                episode = (e + 1) * config["eval_freq"]
                epsilon = epsilon_trace[min(episode, len(epsilon_trace) - 1)]

                row = [schedule, e + 1, episode, round(epsilon, 6)]
                row += [round(float(mean[i]), 6) for i in range(n_agents)]
                for i in range(n_agents):
                    row += [round(q_tables[i][str((0, a))], 6) for a in range(n_actions)]

                writer.writerow(row)
                n_rows += 1

    if output:
        print(f"Wrote {n_rows} rows ({len(SCHEDULES)} schedules) to {path}\n")
    return path


def visualise_schedule_comparison(schedule_results, agent=0, every=8):
    """
    Print and plot the exploration schedules and the resulting Q-value trajectories

    The epsilon trace spans the full width of the two Q-value panels below it.

    :param schedule_results (Dict[str, Dict]): output of train_schedule for each schedule
    :param agent (int): index of the agent whose Q-values are plotted
    :param every (int): print every n-th evaluation point in the table
    """
    # numeric view of the curves plotted below
    q = {s: np.array([[qts[agent][str((0, a))] for a in range(2)]
                      for qts in schedule_results[s]["eval_q_tables"]]) for s in SCHEDULES}
    n_evals = len(next(iter(q.values())))
    print(f"Agent {agent + 1} Q-values per schedule (every {every}th evaluation)")
    print(f"{'Eval':>6}" + "".join(f"{s[:3] + ' Q(D)':>11}" for s in SCHEDULES)
                         + "".join(f"{s[:3] + ' Q(C)':>11}" for s in SCHEDULES))
    for e in range(0, n_evals, every):
        row = f"{e + 1:>6}"
        for act in (1, 0):
            for s in SCHEDULES:
                row += f"{q[s][e, act]:>11.3f}"
        print(row)
    print()

    fig = plt.figure(figsize=(FIG_WIDTH * 2.4, FIG_HEIGHT * 3.4))
    gs = fig.add_gridspec(2, 2, wspace=FIG_WSPACE, hspace=0.45)
    ax_eps = fig.add_subplot(gs[0, :])
    ax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for s in SCHEDULES:
        ax_eps.plot(schedule_results[s]["epsilon_trace"], label=SCHEDULE_LABELS[s])
        q_values = np.array([[qts[agent][str((0, a))] for a in range(2)]
                             for qts in schedule_results[s]["eval_q_tables"]])
        ax[0].plot(q_values[:, 1], label=SCHEDULE_LABELS[s])
        ax[1].plot(q_values[:, 0], label=SCHEDULE_LABELS[s])

    ax_eps.set_xlabel("Episodes")
    ax_eps.set_ylabel("epsilon")
    ax_eps.legend()

    ax[0].axhline(1.0, ls="--", color="k", alpha=0.4)
    ax[1].axhline(0.0, ls="--", color="k", alpha=0.4)
    ax[0].set_title(f"Agent {agent + 1}: Q(Defect)")
    ax[1].set_title(f"Agent {agent + 1}: Q(Cooperate)")
    for a in ax:
        a.set_xlabel("Evaluations")
        a.set_ylabel("Q-value")
        a.legend()

    plt.show()


if __name__ == "__main__":
    from matrix_game import create_pd_game
    from train_iql import CONFIG

    env = create_pd_game()
    schedule_results = {s: train_schedule(env, CONFIG, s) for s in SCHEDULES}

    # written before the plots, which block until their windows are closed
    save_schedule_csv(schedule_results, CONFIG)

    visualise_schedule_comparison(schedule_results)