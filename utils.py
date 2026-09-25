import csv
import os

import matplotlib.pyplot as plt
import numpy as np

FIG_WIDTH = 5
FIG_HEIGHT = 2
FIG_ALPHA = 0.2
FIG_WSPACE = 0.3
FIG_HSPACE = 0.2

ACTION_LABELS = {0: "Cooperate", 1: "Defect"}

OUTPUT_DIR = "outputs"
EVAL_CSV_NAME = "matrix_game_iql.csv"
SCHEDULE_CSV_NAME = "matrix_game_iql_schedules.csv"


def visualise_q_tables(q_tables):
    for i, q_table in enumerate(q_tables):
        print(f"Q-table for Agent {i + 1}:")
        for a in range(2):
            print(f"Q({ACTION_LABELS[a]}) = {q_table[str((0, a))]:.2f}")
        print()


def visualise_evaluation_returns(means, stds, every=4):
    """
    Print and plot evaluation returns

    :param means (List[List[float]]): mean evaluation returns for each agent
    :param stds (List[List[float]]): standard deviation of evaluation returns for each agent
    :param every (int): print every n-th evaluation point in the table
    """
    n_agents = len(means[0])
    n_evals = len(means)

    # numeric view of the curves plotted below
    print(f"Evaluation returns ({n_evals} evaluations, every {every}th shown)")
    print(f"{'Eval':>6}{'A1 mean':>10}{'A1 std':>9}{'A2 mean':>10}{'A2 std':>9}")
    for e in range(0, n_evals, every):
        print(f"{e + 1:>6}{means[e][0]:>10.3f}{stds[e][0]:>9.3f}"
              f"{means[e][1]:>10.3f}{stds[e][1]:>9.3f}")
    print()

    fig, ax = plt.subplots(nrows=1, ncols=n_agents, figsize=(FIG_WIDTH * 2, FIG_HEIGHT * n_agents))

    colors = ["b", "r"]
    for i, color in enumerate(colors):
        ax[i].plot(range(n_evals), [mean[i] for mean in means], label=f"Agent {i+1}", color=color)
        ax[i].fill_between(range(n_evals), [mean[i] - std[i] for mean, std in zip(means, stds)],
                           [mean[i] + std[i] for mean, std in zip(means, stds)], alpha=FIG_ALPHA, color=color)
        ax[i].set_xlabel("Evaluations")
        ax[i].set_ylabel("Evaluation return")
    fig.legend()
    fig.subplots_adjust(hspace=FIG_HSPACE)

    plt.show()

def visualise_q_convergence(eval_q_tables, env, savefig=None, every=8):
    """
    Print and plot q_table convergence
    :param eval_q_tables (List[List[Dict[Act, float]]]): q_tables of both agents for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param savefig (str): path to save figure
    :param every (int): print every n-th evaluation point in the table
    """
    assert hasattr(env, "payoff")
    payoff = np.array(env.payoff)
    n_actions = 2
    n_agents = 2
    assert payoff.shape == (n_actions, n_actions, n_agents), "Payoff matrix must be 2x2x2 for 2x2 PD game"
    # (n_evals, n_agents, n_actions)
    q_tables = np.array(
            [[[q_table[str((0, act))] for act in range(n_actions)] for q_table in q_tables] for q_tables in eval_q_tables]
    )

    # numeric view of the curves plotted below
    print(f"Q-values (every {every}th evaluation)")
    print(f"{'Eval':>6}{'A1 Q(C)':>10}{'A1 Q(D)':>10}{'A2 Q(C)':>10}{'A2 Q(D)':>10}")
    for e in range(0, len(q_tables), every):
        print(f"{e + 1:>6}{q_tables[e, 0, 0]:>10.3f}{q_tables[e, 0, 1]:>10.3f}"
              f"{q_tables[e, 1, 0]:>10.3f}{q_tables[e, 1, 1]:>10.3f}")
    print()

    fig, ax = plt.subplots(nrows=n_agents, ncols=n_actions, figsize=(n_actions * FIG_WIDTH * 1.2, FIG_HEIGHT * n_agents * 1.75))

    for i in range(n_agents):
        max_payoff = payoff[:, :, i].max()
        min_payoff = payoff[:, :, i].min()

        for act in range(n_actions):
            # plot max Q-values
            if i == 0:
                max_r = payoff[act, :, i].max()
                max_label = rf"$max_b Q(a, b)$"
                q_label = rf"$Q(a_{act}, \cdot)$"
            else:
                max_r = payoff[:, act, i].max()
                max_label = rf"$max_a Q(a, b_{act})$"
                q_label = rf"$Q(\cdot, b_{act})$"
            ax[i, act].axhline(max_r, ls='--', color='r', alpha=0.5, label=max_label)

            # plot respective Q-values
            q_values = q_tables[:, i, act]
            ax[i, act].plot(q_values, label=q_label)

            # axes labels and limits
            ax[i, act].set_ylim([min_payoff - 0.05, max_payoff + 0.05])
            ax[i, act].set_xlabel(f"Evaluations")
            if i == 0:
                ax[i, act].set_ylabel(fr"$Q(a_{act})$")
            else:
                ax[i, act].set_ylabel(fr"$Q(b_{act})$")

            ax[i, act].legend(loc="upper center")

    fig.subplots_adjust(wspace=FIG_WSPACE, hspace=FIG_HSPACE * 2)

    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")

    plt.show()


def save_evaluation_csv(means, stds, eval_q_tables, config, epsilons=None,
                        path=None, output=True):
    """
    Write evaluation returns and Q-values of a training run to a CSV file

    One row per evaluation point; the episode of evaluation e is (e + 1) * eval_freq.

    :param means (List[List[float]]): mean evaluation returns for each agent
    :param stds (List[List[float]]): standard deviation of evaluation returns for each agent
    :param eval_q_tables (List[List[Dict[Act, float]]]): q_tables of both agents for each evaluation
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param epsilons (List[float]): training epsilon at each evaluation point
    :param path (str): path of the CSV file to write
    :param output (bool): flag whether the written path should be printed
    :return (str): path of the written CSV file
    """
    if path is None:
        path = os.path.join(OUTPUT_DIR, EVAL_CSV_NAME)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    n_agents = len(means[0])
    n_actions = len(ACTION_LABELS)

    header = ["evaluation", "episode", "timestep", "epsilon"]
    for i in range(n_agents):
        header += [f"agent{i + 1}_return_mean", f"agent{i + 1}_return_std"]
    for i in range(n_agents):
        header += [f"agent{i + 1}_q_{ACTION_LABELS[a].lower()}" for a in range(n_actions)]
    header += [f"agent{i + 1}_greedy_action" for i in range(n_agents)]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for e, (mean, std, q_tables) in enumerate(zip(means, stds, eval_q_tables)):
            episode = (e + 1) * config["eval_freq"]
            q_values = [[q_tables[i][str((0, a))] for a in range(n_actions)]
                        for i in range(n_agents)]

            row = [e + 1, episode, episode * config["ep_length"],
                   "" if epsilons is None else round(epsilons[e], 6)]
            for i in range(n_agents):
                row += [round(float(mean[i]), 6), round(float(std[i]), 6)]
            for i in range(n_agents):
                row += [round(q_values[i][a], 6) for a in range(n_actions)]
            row += [ACTION_LABELS[int(np.argmax(q_values[i]))] for i in range(n_agents)]

            writer.writerow(row)

    if output:
        print(f"Wrote {len(means)} evaluation points to {path}\n")
    return path