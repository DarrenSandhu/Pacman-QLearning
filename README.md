# Q-Learning Agent For Pacman

This project implements an **Q-Learning Agent** for Pacman, using value iteration through supervised learning to determine the optimal policy for Pacman's actions in the game grid.

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
# Run with UI
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid 

# Run without UI (quiet mode)
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid -q

```

### Explanation of Command:
- `QLearnAgent`: Specifies the Q-Learning-based agent for controlling Pacman’s actions.
- `-p`: Directs pacman.py to locate and activate the specified agent (in this case, QLearnAgent).
- `-x`: Specifies the number of training games (in this example, 2000), where the agent learns from its environment without focusing on scoring.
- `-n`: Indicates the total number of games to run (here, 2010). This allows the agent to use what it learned during the training games in the final 10 games.
- `-l`: Chooses the grid layout (in this example, smallGrid), which defines the environment Pacman will navigate
- `-q`: Runs the game in "quiet mode" without the graphical UI, which speeds up the simulation for large-scale training.

## How the Q-Learning Agent Works

- **State Representation**: Each state uniquely represents the configuration of Pacman, the ghosts, the food, and the grid layout. This representation allows the agent to differentiate between scenarios and make decisions based on the environment it perceives at any given time.
- **Rewards and Transitions**: The agent receives rewards for certain actions, such as eating food or avoiding ghosts, which are vital for learning. Rewards are tied to state-action pairs and guide the agent’s behavior by encouraging beneficial actions. Transitions between states follow game rules, updating the agent’s position based on its chosen actions and adjusting the game environment accordingly.
- **Q-Value Calculation (Value Iteration)**: The agent calculates the Q-value, or expected future reward, for each state-action pair. Using the Q-learning algorithm, it iteratively updates these Q-values by factoring in immediate rewards and estimating the future rewards from subsequent states. Over time, this enables the agent to converge on optimal Q-values that guide it toward maximum cumulative rewards.
- **Policy Execution**: With the Q-values derived, the agent follows a policy that selects actions with the highest Q-values in each state. This policy is the agent’s learned strategy, maximizing rewards and ensuring that it makes the best possible decisions based on prior experience.

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

