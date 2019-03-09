# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Gets n of food left
        food_left = 0
        for i in newFood:
            for j in i:
                food_left += int(j)
        # Searches for the closest ghost
        nearest_ghost_dist = 0
        if newGhostStates:
            ghost_distances_list = []
            for ghost in newGhostStates:
                ghost_distances_list.append(manhattanDistance(ghost.getPosition(), newPos))
                nearest_ghost_dist = min(ghost_distances_list)
        if nearest_ghost_dist == 0:
            return -sys.maxint
        # Searches for the closest food
        nearest_food_dist = 0
        if food_left > 0:
            food_distances_list = []
            for x, m in enumerate(newFood):
                for y, n in enumerate(m):
                    if n:
                        food_distances_list.append(manhattanDistance(newPos, (x, y)))
            nearest_food_dist = min(food_distances_list)
        return - (10/nearest_ghost_dist + 5*nearest_food_dist +200*food_left)

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def minimax(gameState, depth=0, agent=self.index):
            agent = agent % gameState.getNumAgents()
            mm_action, mm_value = None, (-sys.maxint if agent == 0 else sys.maxint)
            n_agent, n_depth = agent + 1, depth + 1
            actions = gameState.getLegalActions(agent)
            # Final state (base case)
            if agent == 0 and depth == self.depth or (len(actions) == 0):
                return self.evaluationFunction(gameState), None
            # Recursive step
            # If agent is pacman: them max
            if agent == 0:
                for action in actions:
                    n_value, n_action = minimax(gameState.generateSuccessor(agent, action), n_depth, n_agent)
                    if n_value > mm_value:
                        mm_action, mm_value = action, n_value
                return mm_value, mm_action
            # If agent is ghost: then min
            else:
                for action in actions:
                    n_value, n_action = minimax(gameState.generateSuccessor(agent, action), depth, n_agent)
                    if n_value < mm_value:
                        mm_action, mm_value = action, n_value
                return mm_value, mm_action
        dropped, f_action = minimax(gameState)
        return f_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta_pruning(gameState, depth=0, agent=self.index, alpha=-sys.maxint, beta=sys.maxint):
            agent = agent % gameState.getNumAgents()
            ab_action, ab_value = None, (-sys.maxint if agent == 0 else sys.maxint)
            n_agent, n_depth = agent + 1, depth + 1
            actions = gameState.getLegalActions(agent)
            # Final state (base case)
            if agent == 0 and depth == self.depth or (len(actions) == 0):
                return self.evaluationFunction(gameState), None
            # Recursive step
            # If agent is pacman: them max
            if agent == 0:
                for action in actions:
                    successor = gameState.generateSuccessor(agent, action)
                    n_value, n_action = alpha_beta_pruning(successor, n_depth, n_agent, alpha, beta)
                    if n_value > ab_value:
                        ab_action, ab_value = action, n_value
                    if ab_value > beta:
                        return ab_value, ab_action
                    alpha = max(alpha, ab_value)
                return ab_value, ab_action
            # If agent is ghost: then min
            else:
                for action in actions:
                    successor = gameState.generateSuccessor(agent, action)
                    n_value, n_action = alpha_beta_pruning(successor, depth, n_agent, alpha, beta)
                    if n_value < ab_value:
                        ab_action, ab_value = action, n_value
                    if ab_value < alpha:
                        return ab_value, ab_action
                    beta = min(beta, ab_value)
                return ab_value, ab_action
        dropped, f_action = alpha_beta_pruning(gameState)
        return f_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectimax(gameState, depth=0, agent=self.index):
            agent = agent % gameState.getNumAgents()
            em_action, em_value = None, (-sys.maxint if agent == 0 else 0)
            n_agent, n_depth = agent + 1, depth + 1
            actions = gameState.getLegalActions(agent)
            # Final state (base case)
            if agent == 0 and depth == self.depth or (len(actions) == 0):
                return self.evaluationFunction(gameState), None
            # Recursive step
            # If agent is pacman: them max
            if agent == 0:
                for action in actions:
                    n_value, n_action = expectimax(gameState.generateSuccessor(agent, action), n_depth, n_agent)
                    if n_value > em_value:
                        em_action, em_value = action, n_value
                return em_value, em_action
            # If agent is ghost: then expect
            else:
                times = 0
                for action in actions:
                    n_value, n_action = expectimax(gameState.generateSuccessor(agent, action), depth, n_agent)
                    em_value += n_value
                    times += 1
                return (float(em_value)/times), em_action
        dropped, f_action = expectimax(gameState)
        return f_action



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    current_pos = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()
    current_ghost_states = currentGameState.getGhostStates()
    current_scared_timers = [ghostState.scaredTimer for ghostState in current_ghost_states]

    # Gets n of food left
    food_left = 0
    for i in current_food:
        for j in i:
            food_left += int(j)
    # Calculates shortest scare time
    shortest_scared_timer = min(current_scared_timers)
    # Searches for the closest ghost
    nearest_ghost_dist = 0
    if current_ghost_states:
        ghost_distances_list = []
        for ghost in current_ghost_states:
            ghost_distances_list.append(manhattanDistance(ghost.getPosition(), current_pos))
            nearest_ghost_dist = min(ghost_distances_list)
    if nearest_ghost_dist == 0:
        return -sys.maxint
    # Depending if the ghost is scared or not, distance is changed.
    if shortest_scared_timer == 0:
        nearest_ghost_dist = -10/nearest_ghost_dist
    else:
        nearest_ghost_dist = 3*nearest_ghost_dist
    # Searches for the closest food
    nearest_food_dist = 0
    if food_left > 0:
        food_distances_list = []
        for x, m in enumerate(current_food):
            for y, n in enumerate(m):
                if n:
                    food_distances_list.append(manhattanDistance(current_pos, (x, y)))
        nearest_food_dist = min(food_distances_list)
    # Returns the same as the first evaluation but takes into consideration whether a ghost is scared
    # the time remaining of the ghost being scared and the score of the game.
    return -(200*food_left - nearest_ghost_dist + 5*nearest_food_dist - 100*(shortest_scared_timer + currentGameState.getScore()))

# Abbreviation
better = betterEvaluationFunction

