# Markov Decision Process For Pacman

This project implements an **Q-Learning Agent** for Pacman, using value iteration through reinforcement learning to determine the optimal policy for Pacman's actions in the game grid.

## Features

- **Q-Learning Agent**: The agent models Pacman's world as a grid of states with rewards and transitions.
- **Value Iteration**: Uses value iteration to compute the optimal policy for each state, maximizing rewards over time.
- **Gridworld Support**: The agent can be run on various grid layouts.
- **Pacman Simulation**: This agent is compatible with the Pacman simulation environment and can be used with various layouts and configurations of the game.

  
## Prerequisites

- **Python 3** (or Python 3.x if applicable)
- **Pacman AI Project**: Make sure you have the full Pacman project environment.
  
## Running the Agent

To run the **QLearnAgent** in the Pacman environment, follow these steps:

1. **Navigate to the Project Directory**: First, ensure you're in the correct directory where the Pacman code is located:

```bash
cd pacman-cw2  # Navigate to the directory containing pacman.py
```

2. **Run the Agent**: Use the following command:

```bash
python3 pacman.py --pacman QLearnAgent
```

### Explanation of Command:
- `QLearnAgent`: Use the QLearn-based agent for decision-making.
- `--pacman`: Instructs pacman.py to search for and execute the specified agent (QLearnAgent in this case) during gameplay.

## How the MDPAgent Works

- **State Representation**: Each state represents a unique configuration of Pacman, the ghosts, and the grid.
- **Rewards and Transitions**: The agent receives rewards based on actions (eating food, avoiding ghosts) and transitions between states according to game rules.
- **Value Iteration**: The agent iteratively calculates the value of each state to find the optimal policy for maximum reward.
- **Policy Execution**: After value iteration, the agent follows the policy to make decisions during gameplay.

## Future Improvements

- **Ghost Avoidance**: Implement improved ghost prediction strategies for better ghost evasion.
- **Encourage Ghost Chase During Capsule Power Up**: When a power-up capsule is consumed, the agent should be encouraged to chase and eat ghosts.
- **Reward Tuning**: Adjust the reward values for specific events (e.g., eating food or avoiding ghosts) to improve agent behavior.

## License
This project is licensed for educational purposes under the following conditions:

- You may not distribute or publish solutions to these projects.
- You must retain this notice in all copies or substantial portions of the code.
- You must provide clear attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

For detailed information, refer to the [LICENSE](LICENSE) file.

