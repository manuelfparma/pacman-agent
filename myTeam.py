# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint

import itertools
import time

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveMinimaxAgent', second='DefensiveMinimaxAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


###################
# Original Agents #
###################

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        # offensive score: -100 * #food_left - min_dist_to_food
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

##################
### OUR AGENTS ###
##################
# This class is to contain features and weights for offensive playing
class OfensiveFeatures():
    """
    Our additions to the OffensiveReflexAgent:
    - Take into account if an enemy ghost is near, as we don't want to be eaten
    - If pac-man has eaten a good number of food, return home to obtain score
    - If the pac-man is in danger, try to get a capsule to eat enemy ghosts
    """
    def get_features(self, agent, game_state, action):
        successor = agent.get_successor(game_state, action)
        features = self.get_current_features(agent, successor) 
        return features

    def get_current_features(self, agent, game_state):
        features = util.Counter()
        my_state = game_state.get_agent_state(agent.index)
        my_pos = my_state.get_position()
        food_list = agent.get_food(game_state).as_list()

        # same code as the default Offensive Agent
        features['food_score'] = len(food_list)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([agent.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # check if enemies are close
        if my_state.is_pacman:
            enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
            close_enemies = [a for a in enemies if a.get_position() is not None]
            if len(close_enemies) > 0:
                features['in_danger'] = 1
                dists = [agent.get_maze_distance(my_pos, a.get_position()) for a in close_enemies]
                min_dist = min(dists)
                if min_dist == 0:
                    features['in_danger'] = 100
                features['enemy_distance'] = min_dist
                # try to get close to a capsule
                capsule_list = agent.get_capsules(game_state)
                if len(capsule_list) > 0:
                    min_distance = min([agent.get_maze_distance(my_pos, capsule) for capsule in capsule_list])
                    features['distance_to_food'] = 10 * min_distance

        features['food_eaten'] = my_state.num_carrying + my_state.num_returned

        # check if we have eaten lots of food
        if my_state.num_carrying > 1:
            features['distance_home'] =  my_state.num_carrying * agent.get_maze_distance(my_pos, agent.start)

        return features

    def get_weights(self):
        return {'food_score': -100, 
                'distance_to_food': -10, 
                'in_danger': -10, 
                'enemy_distance': 1, 
                'distance_home': -10, 
                'food_eaten': 1000
                }

# This class is to contain features and weights for defensive playing
class DefensiveFeatures():
    '''
    Copied from DefensiveReflexAgent.
    We added the following features:
    - The Defensive Agent now wants to get away from opponent Pac-Man with power-ups (can eat ghosts)
    - To make the ghost patrol food, it does one of the following:
        * Goes to a random food location from our team
        * When it arrives there, choose a new random point
        * If a food was eaten recently, go there (because a Pac-Man is probably nearby)
    '''
    def __init__(self):
        self.food_objective = None

    def get_features(self, agent, game_state, action):
        successor = agent.get_successor(game_state, action)
        features = self.get_current_features(agent, successor)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(agent.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_current_features(self, agent, game_state):
        features = util.Counter()

        my_state = game_state.get_agent_state(agent.index)
        my_pos = my_state.get_position()
        my_food = agent.get_food_you_are_defending(game_state).as_list()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [agent.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            if my_state.scared_timer == 0:
                features['invader_distance'] = min(dists)
            else:
                features['invader_distance'] = -min(dists)

        # guarding the food: choose a random food point and go there
        if len(my_food) > 0:
            # if no objective is choosen, get one
            if self.food_objective == None or not self.food_objective in my_food:
                self.food_objective = random.sample(my_food, 1)[0]

            # check if the enemy ate food recently
            eaten_food = self.__find_eaten_food(agent, game_state)
            if eaten_food != None:
                self.food_objective = eaten_food

            food_distance = agent.get_maze_distance(my_pos, self.food_objective)
            # if we have arrived at our objective, go to another point
            if food_distance == 0:
                new_objective = random.sample(my_food, 1)[0]
                food_distance = agent.get_maze_distance(my_pos, new_objective)
                self.food_objective = new_objective

            features['distance_to_food'] = food_distance

        return features

    def get_weights(self):
        return {'num_invaders': -1000, 
                'on_defense': 100, 
                'invader_distance': -100, 
                'stop': -100, 
                'reverse': -2,
                'distance_to_food': -10,
                }

    # if the food on our side was been reduced from the previous state
    # that means the enemy pacman has eaten a food, and we want to go
    # to that food position
    def __find_eaten_food(self, agent, game_state):
        predecessor = agent.get_previous_observation()

        current_food = agent.get_food_you_are_defending(game_state).as_list()

        if predecessor != None:
            previous_food = agent.get_food_you_are_defending(predecessor).as_list()
        else:
            previous_food = current_food

        if len(previous_food) > len(current_food):
            difference = list(set(previous_food) - set(current_food))
            return difference[0]
        
        return None

# Generic Minimax Agent
class MinimaxAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.depth = 2
        self.print_time = False # change to True if you want warnings when decision times are over 1 second

    # get successor but adapted for any index
    def get_successor_idx(self, game_state, action, agentIndex):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(agentIndex, action)
        pos = successor.get_agent_state(agentIndex).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(agentIndex, action)
        else:
            return successor

    # This function checks Pac-Man actions and find the one that give the highest score
    # Returns a tuple with the best action and the score it generates
    def __minimaxPacmanMax(self, game_state, depth, alpha, beta):
        # if the game is over, just return the score
        if game_state.is_over():
            return ('Stop', self.evaluate(game_state, 'Stop'))

        maxScore = -999999
        maxAction = None

        # for each pacman action we check what the ghosts may do
        for action in game_state.get_legal_actions(self.index):
            if depth > 1:
                nextState = self.get_successor_idx(game_state, action, self.index)
                score = self.__minimaxGhostsMin(nextState, depth, alpha, beta)
            else:
                score = self.evaluate(game_state, action)
            if score > maxScore:
                maxScore = score
                maxAction = action
            # check against the beta
            if score > beta:
                break
            # update the alpha
            alpha = max(alpha, score)

        return (maxAction, maxScore)

    # This function checks possible moves for the ghosts and choose the one with the least score
    # Returns the minimum score
    def __minimaxGhostsMin(self, game_state, depth, alpha, beta):    
        # if the game is over, just return the score
        if game_state.is_over():
            return self.evaluate(game_state, 'Stop')
        
        enemies = [i for i in self.get_opponents(game_state)]
        close_enemies = [i for i in enemies if game_state.get_agent_state(i).get_position() is not None]

        # if no enemies are close, just go to next minimax depth or finish
        if len(close_enemies) == 0:
            score = self.__minimaxPacmanMax(game_state, depth-1, alpha, beta)[1]

        # First, we create a list with all possible ghost actions combinations
        movesList = list()
        for enemyIndex in close_enemies:
            movesList.append(game_state.get_legal_actions(enemyIndex))
        # itertool.product takes a list of list and builds the possible combinations
        possibleMoves = list(itertools.product(*movesList))

        minScore = 999999

        # Then, for each possible combination we calculate its evaluation and find the minimum
        for moves in possibleMoves:
            nextState = game_state
            # generate the next possible state
            for idx, enemyIndex in enumerate(close_enemies):
                if game_state.is_over():
                    break
                nextState = self.get_successor_idx(game_state, moves[idx], enemyIndex)
            # evaluate the next state
            score = self.__minimaxPacmanMax(nextState, depth-1, alpha, beta)[1]
            # check if it is the minimum
            minScore = min(minScore, score)
            # check against the alpha
            if score < alpha:
                break
            # update the beta
            beta = min(beta, score)

        return minScore
    
    def choose_action(self, game_state):
        start = time.time()
        # we start the minimax search with Pac-Man/Max
        result = self.__minimaxPacmanMax(game_state, self.depth, -999999, 999999)
        elapsed = time.time() - start
        if self.print_time and elapsed >= 1.0:
            print(f'eval time for agent {self.index}: {elapsed:.4f} - score: {result[1]}')
        return result[0]

# Check OfensiveFeatures class for description of features
class OffensiveMinimaxAgent(MinimaxAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.offensive_features = OfensiveFeatures()

    def get_features(self, game_state, action):
        return self.offensive_features.get_features(self, game_state, action)

    def get_weights(self, game_state, action):
        return self.offensive_features.get_weights()

# Check DefensiveFeatures class for description of features
class DefensiveMinimaxAgent(MinimaxAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.defensive_features = DefensiveFeatures()

    def get_features(self, game_state, action):
        return self.defensive_features.get_features(self, game_state, action)

    def get_weights(self, game_state, action):
        return self.defensive_features.get_weights()
      
