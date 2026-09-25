# Multi-Agent Reinforcement Learning HW1 - Independent Q-Learning

**Home Assignment (Grade 100, M.Sc. Data Science, HIT). Independent Q-Learning (IQL) from scratch: two agents, each learning its own Q-function with no model of the other, trained on the one-shot Prisoner's Dilemma. The question is what independent learners converge to when cooperation pays more but defection is individually rational.**

## Headline Results
- **Mutual defection:** both agents converge to (Defect, Defect), the game's only Nash equilibrium. Greedy evaluation returns exactly **1.00** for each — although mutual cooperation would pay 3.
- **The Q-values agree:** Q(Defect) ≈ 1.0 and Q(Cooperate) ≈ 0.05 for both agents.
- **Exploration changes the path, not the destination:** linear, exponential and constant ε-schedules all reach the same equilibrium.

## Key Features
- **IQL from Scratch:** ε-greedy action selection and the independent Q-learning update.
- **Learning Dynamics:** returns and Q-value convergence tracked across 49 evaluation points over 20,000 episodes.
- **Exploration-Schedule Extension:** three ε-schedules compared under the same seed, with all results exported to CSV.

## Repository Structure
- `Multi_Agent_Reinforcement_Learning_HW1.ipynb`: Full solution notebook — training, Q-tables, plots and analysis (explanations in Hebrew).
- `iql.py`: The IQL agent — ε-greedy action selection and the Q-learning update.
- `train_iql.py`: Training and evaluation loop.
- `schedule_experiment.py`: Comparison of three ε-schedules — linear, exponential and constant.
- `matrix_game.py`: The Prisoner's Dilemma matrix-game environment.
- `utils.py`: Plotting and CSV helpers.
- `matrix_game_iql.csv`: Evaluation results of the main run.
- `matrix_game_iql_schedules.csv`: Evaluation results of the three ε-schedules.
- `requirements.txt`: Dependencies.
- `Assignment_1.pdf`: Original assignment instructions.

## Source
The assignment is the Tuesday exercise, *Tabular Multi-Agent Reinforcement Learning*, from [marl-book-exercises](https://github.com/marl-book/marl-book-exercises) — designed for the [Barcelona Summer School 2024 on Multi-Agent Reinforcement Learning](https://iiia.csic.es/en-us/marl-course/) and based on the textbook [*Multi-Agent Reinforcement Learning: Foundations and Modern Approaches*](https://marl-book.com/).
