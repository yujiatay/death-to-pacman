# Death to Pac-Man: Ghost Revolution with Multi-agent Deep Reinforcement Learning 

## Prerequisites
1. Python 3 and higher. 

## Getting started
- To install, run `pip install -r requirements.txt`
- To begin training with GUI, run `python train.py --display`

## Usage
| Command-line option | Purpose |
|---------------------|---------|
|`--max-episode-len`  | Maximum length of each episode (default: 100)|
|`--num-episodes`     | Total number of training episodes (default: 200000)|
|`--num-adversaries`  | Number of ghost agents in the environment (default: 2)|
|`--good-policy`      | Algorithm used for Pac-Man agent (default: `ddpg`, options: `ddpg` or `maddpg`)|
|`--adv-policy`       | Algorithm used for Ghost agents (default: `maddpg`, options: `ddpg` or `maddpg`)|
|`--lr`               | Learning rate for Adam optimizer (default: `1e-2`)|
|`--gamma`            | Discount factor (default: `0.95`)|
|`--batch-size`       | Batch size (default: `1024`)|
|`--save-dir`         | Directory where training state and model will be saved (default: `"./save_files/"`)|
|`--save-rate`        | Model is saved every `x` episodes (default: `1000`)|
|`--restore`          | Restore training from last training checkpoint (default: `False`)|
|`--display`          | Displays the GUI (default: `False`)|
|`--load-dir`         | Directory where training state and model are loaded from (default: `""`)|
|`--load`             | Only loads model if this is set to `True` (default: `False`)|
|`--load-episode`     | Loads a model tagged to a particular episode (default: `0`)|
|`--pacman_obs_type`  | Observation space for Pac-Man agent (default: `partial_obs`, options: `partial_obs` or `full_obs`)|
|`--ghost_obs_type`   | Observation space for Ghost agents (default: `full_obs`, options: `partial_obs` or `full_obs`)|
|`--partial_obs_range`| Range for partial observation space, if chosen (default: `3`) e.g. 3x3, 5x5, 7x7...|
|`--shared_obs`       | Include same features in observation spaces of both Pac-Man and Ghost agents (default: `False`)|
|`--astarSearch`      | Factor step distance between Pac-Man and Ghost into reward and observation of agents (default: `False`)|
|`--astartAlpha`      | Multiplier for penalizing/rewarding agents using increase/decrease in step distance (default: `1`)|

## Authors
- [Anuj Sanjay Patel](https://github.com/anujsp2797)
- [Franklin Leong](https://github.com/FranklinLeong)
- [Marvin Tan](https://github.com/marvintxd)
- [Tay Yu Jia](https://github.com/yujiatay)
- [Teo Wei Zheng](https://github.com/teowz46)
- [Xiu Ziheng](https://github.com/Cary-Xx)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details

## Acknowledgements
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://github.com/openai/maddpg)
- [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs)
- [OpenAI Gym Environment of Berkeley AI Pacman](https://github.com/sohamghosh121/PacmanGym)
