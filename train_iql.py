import copy
import random

import gymnasium as gym
import numpy as np

from iql import IQL
from utils import (
    save_evaluation_csv,
    visualise_q_tables,
    visualise_q_convergence,
    visualise_evaluation_returns,
)
from matrix_game import create_pd_game


CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "total_eps": 20000,
    "ep_length": 1,
    "eval_freq": 400,
    "lr": 0.05,
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
}


def iql_eval(env, config, q_tables, eval_episodes=500, output=True):
    """
    Evaluate configuration of independent Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict[Act, float]]): Q-tables mapping actions to Q-values for each agent
    :param eval_episodes (int): number of evaluation episodes
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agents = IQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["eval_epsilon"],
    )
    eval_agents.q_tables = copy.deepcopy(q_tables)

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print("EVALUATION RETURNS:")
        print(f"\tAgent 1: {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"\tAgent 2: {mean_return[1]:.2f} ± {std_return[1]:.2f}")
    return mean_return, std_return


def evaluation_epsilons(env, config, n_evals):
    """
    Training epsilon at each evaluation point

    Queried from the agents' own schedule_hyperparameters, so the values match
    the schedule that was actually used during training.

    :param env (gym.Env): environment the run was executed on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param n_evals (int): number of evaluation points of the run
    :return (List[float]): training epsilon at each evaluation point
    """
    probe_agents = IQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )
    max_steps = config["total_eps"] * config["ep_length"]

    epsilons = []
    for e in range(n_evals):
        timestep = (e + 1) * config["eval_freq"] * config["ep_length"]
        probe_agents.schedule_hyperparameters(timestep, max_steps)
        epsilons.append(probe_agents.epsilon)
    return epsilons


def train(env, config, output=True):
    """
    Train and evaluate independent Q-learning in env with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (List[List[float]], List[List[float]], List[Dict[Act, float]]):
    """
    agents = IQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            episodic_return += rewards
            obss = n_obss

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            eval_count = eps_num // config["eval_freq"]
            show = output and (eval_count % 10 == 1)
            if show:
                print(f"Evaluation {eval_count} (episode {eps_num}):")
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=show
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables,
    )


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    env = create_pd_game()
    # env = gym.make("lbforaging:Foraging-5x5-2p-1f-v3")
    evaluation_return_means, evaluation_return_stds, eval_q_tables, q_tables = train(
        env, CONFIG
    )
    # written before the plots, which block until their windows are closed
    save_evaluation_csv(
        evaluation_return_means,
        evaluation_return_stds,
        eval_q_tables,
        CONFIG,
        epsilons=evaluation_epsilons(env, CONFIG, len(evaluation_return_means)),
    )

    visualise_q_tables(q_tables)
    visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds)
    visualise_q_convergence(eval_q_tables, env)