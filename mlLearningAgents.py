# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import math
import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.state = state



class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.5, # Encourage exploration at the beginning
                 gamma: float = 0.9,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0  # Count the number of games we have played
        self.q_values_dict = {} # Initialize q-values table
        self.transitions = []  # Initialize transitions list
        self.state_action_counts = {}  # Initialize counts table

       


    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value
    
    def setGamma(self, value: float):
        self.gamma = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Compute the reward based on the information from the start state
        and the end state. 
        Where the reward structure encourages Pacman to eat food, avoid ghosts, 
        avoid walls, win the game, and avoid losing the game.


        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        pacman_pos = endState.getPacmanPosition() # Get Pacmans position
        all_ghosts = endState.getGhostPositions() # Get all the ghosts positions
    
        # Check if Pacman has lost the game
        if endState.isLose():
            print("Pacman has lost the game\n")
            return -10 # A large negative reward for losing the game
        
        # Check if Pacman has won the game
        if endState.isWin():
            print("Pacman has won the game\n")
            return 20 # A large positive reward for winning the game
            
        
        # Check if Pacman has collided with a wall
        walls = endState.getWalls()
        if walls[pacman_pos[0]][pacman_pos[1]]:
            return -0.1 # A small negative reward for colliding with a wall
    
        
        # Check if Pacman has eaten a food
        if endState.getNumFood() < startState.getNumFood():
            return 5 # A positive reward for eating a food
        
        # Check if Pacman has collided with a ghost
        for ghost_pos in all_ghosts:
            if pacman_pos == ghost_pos:
                return -5 # A large negative reward for colliding with a ghost
            
        # Check if Pacman has moved closer to food
        all_food = endState.getFood().asList()
        for food_pos in all_food:
            start_food_distance = util.manhattanDistance(startState.getPacmanPosition(), food_pos)
            end_food_distance = util.manhattanDistance(endState.getPacmanPosition(), food_pos)
            food_distance_diff = start_food_distance - end_food_distance
            for ghost_pos in all_ghosts:
                start_ghost_food_distance = util.manhattanDistance(ghost_pos, food_pos)
                end_ghost_food_distance = util.manhattanDistance(ghost_pos, food_pos)
                ghost_distance_diff = end_ghost_food_distance - start_ghost_food_distance
                if food_distance_diff > 0 and ghost_distance_diff > 0:
                    return 1 # A small positive reward for moving closer to food
                
                  
        # Check if Pacman has moved to a new empty position
        else:
            return -0.1 # A small negative reward for moving to a new empty position to encourage Pacman to find the food

    
    # Function to set q-value given the state and q-value
    def setQvalue(self, state: GameStateFeatures, action: Directions, q_value):
        """
        Set the Q-value for a given state and action

        Args:
            state: A given state
            action: Proposed action to take
            q_value: The Q-value to set
        """
        try:
            # Try to set the q-value for the given state and action
            self.q_values_dict[(state.state, action)] = q_value
        except KeyError:
            # If the key does not exist, create a new key-value pair for the given state and action with the q-value
            self.q_values_dict.setdefault((state,action), q_value)
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        try:
            # Try to get the q-value for the given state and action
            q_value = self.q_values_dict[(state.state, action)]
        except KeyError:
            # If the key does not exist, create a new key-value pair for the given state and action with the q-value
            self.q_values_dict.setdefault((state.state, action), 0.0)
            q_value = self.q_values_dict[(state.state, action)] 

        return q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum estimated Q-value attainable from the state,
        using the Q-values from the q_values_dict

        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        legal_actions = state.state.getLegalActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
    
        max_q_value = 0.0
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
        
        return max_q_value
    

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # Compute the Q-value for the current state and action
        current_q_value = self.getQValue(state, action)

        # Get learning rate
        learning_rate = self.getAlpha()

        # Get discount factor
        discount_factor = self.getGamma()

        # Compute the maximum Q-value for the next state
        max_q_value = self.maxQValue(nextState)

        # Update the Q-value for the current state and action
        new_q_value = (current_q_value) + (learning_rate * (reward + (discount_factor * max_q_value) - current_q_value))


        self.setQvalue(state, action, new_q_value)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        try:
            self.state_action_counts[(state.state, action)] += 1
        except KeyError:
            self.state_action_counts.setdefault((state.state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.state_action_counts.get((state.state, action), 0)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        Implement an exploration function that encourages exploration at the beginning
        and then gradually decreases the exploration function as the number of games 
        played increases, until it is set to 0.

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """

        # Encourage exploration at the beginning by setting the exploration function to a high value if the counts are 0
        if self.getEpisodesSoFar() < 50:
            if counts == 0:
                return utility + 5
            else:
                return utility
        # Gradually decrease the exploration function as the number of games played increases
        elif self.getEpisodesSoFar() < 500:
            if counts == 0:
                return utility + 1
            else:
                return utility
        # Gradually decrease the exploration function as the number of games played increases
        elif self.getEpisodesSoFar() < 1000:
            if counts == 0:
                return utility + 0.5
            else:
                return utility
        # Once the number of games played exceeds 1000, set the exploration function to 0
        else:
            return utility
        
        
    # Function to get the next state given the current state and action
    def getNextState(self, state: GameState, action: Directions):
        """
        Get the next state given the current state and action
        and return the next state

        Args:
            state: the current state
            action: the action to take

        Returns:
            The next state
        """

        # Get the next state from the current state and action
        next_state = state.generatePacmanSuccessor(action)

        return next_state
    

    # Function to adjust the learning rate based on the number of games played
    def adjustLearningRate(self):
        """
        Adjust the learning rate based on the number of games played,
        to improve the learning process.
        """
        if (self.getEpisodesSoFar() < (self.getNumTraining()/2)):
            self.setAlpha(self.alpha)
        elif (self.getEpisodesSoFar() < self.getNumTraining()):
            self.setAlpha(self.alpha/2)
        else:
            self.setAlpha(0.0005)
    

    # Function to adjust the exploration rate based on the number of games played
    def adjustEpsilon(self):
        """
        Adjust the exploration rate based on the number of games played,
        where the exploration rate is decreased as the number of games played increases.

        This encoruages exploration at the beginning and then gradually decreases the exploration rate,
        until it is set to 0.
        """
        if (self.getEpisodesSoFar() < (self.getNumTraining()/40)):
            self.setEpsilon(self.epsilon)
        elif (self.getEpisodesSoFar() < (self.getNumTraining()/20)):
            self.setEpsilon(0.1)
        elif (self.getEpisodesSoFar() < (self.getNumTraining()/8)):
            self.setEpsilon(0.05)
        elif (self.getEpisodesSoFar() < (self.getNumTraining()/4)):
            self.setEpsilon(0.01)
        elif (self.getEpisodesSoFar() < (self.getNumTraining()/2.7)):
            self.setEpsilon(0.001)
        else:
            self.setEpsilon(0.000001)
            

    # Function to adjust the discount factor based on the number of games played
    def adjustGamma(self):
        """
        Adjust the discount factor based on the number of games played
        """
        if self.getEpisodesSoFar() < 500:
            self.setGamma(0.9)
        elif self.getEpisodesSoFar() < 1000:
            self.setGamma(0.8)
        else:
            self.setGamma(0.5)


    # Function to return the best action that maximises the Q-value given the current state and legal actions
    def doExploitAction(self, game_state_features: GameStateFeatures, legal_actions):
        """
        Choose the action with the highest Q-value to exploit the current state

        Args:
            game_state_features: the current state
            legal_actions: the legal actions that can be taken
        
        Returns:
            The action to take
        """

        best_action = None
        max_q_value = -999999
        update = False # Flag to check if the best action has been found
        for action in legal_actions:
            q_value = self.explorationFn(self.getQValue(game_state_features, action), self.getCount(game_state_features, action)) # Compute the exploration function
            # Select the action with the highest Q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
                update = True
        if update == False:
            print("No Best Action Found\n")
        action = best_action
        
        return action


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        Using the epsilon-greedy algorithm with an exploration function,
        the agent will choose an action to take based on the current state
        and the Q-values for each action.

        After every action, the agent will learn from the current state, action, reward, and next state.

        Args:
            state: the current state

        Returns:
            The action to take
        """

        game_state_features = GameStateFeatures(state) # Get the game state features using wrapper class
        
        # Get legal actions and remove the stop action
        legal_actions = state.getLegalPacmanActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)


        # Use epsilon-greedy exploration
        if util.flipCoin(self.epsilon):
            # Explore: choose a random action
            action = random.choice(legal_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            action = self.doExploitAction(game_state_features, legal_actions)
        
        # Only allow the agent to learn and update q-values during training
        if self.getEpisodesSoFar() < self.getNumTraining():
            # Get the next state
            next_state = self.getNextState(state, action)
            next_state_features = GameStateFeatures(next_state)

            # Compute the reward for the current state and action
            reward = self.computeReward(state, next_state)

            # Learn from the current state, action, reward, and next state
            self.learn(game_state_features, action, reward, next_state_features)

            # Update visitation count for the chosen action
            self.updateCount(game_state_features, action)
            
            # Update the score
            self.score = state.getScore()

            # Update transitions
            self.transitions.append((state, action, reward, next_state))

        return action
    

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        

        # Update q-values and learning rates for the final state only if there are transitions and the number of games played is less than the number of training episodes
        if len(self.transitions) > 0 and self.getEpisodesSoFar() < self.getNumTraining():
            final_state = self.transitions[-1][0] # Get the final state
            final_action = self.transitions[-1][1] # Get the final action
            final_reward = self.transitions[-1][2] # Get the final reward
            final_next_state = self.transitions[-1][3] # Get the final next state
            final_state_features = GameStateFeatures(final_state)
            final_next_state_features = GameStateFeatures(final_next_state)
            self.learn(final_state_features, final_action, final_reward, final_next_state_features) 

            # Adjust the learning rate and epsilon after each game
            self.adjustLearningRate()
            self.adjustEpsilon()

       

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
