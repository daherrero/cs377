# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def search(problem, container):
    """
    The search function will serve to the four different search algorithms but only varying the container type.
    Stores visited locations in visited list.
    Each successor is made up of the successor node (state, action, cost) and a list of the steps to reach that node from
    the start state.
    Gets a node from the container, checks if its the goal state, if it is, returns the second element, the route from the
    start state. If it's not a goal state it gets its successors and push them into the container if they have not already
    been visited. Continues until the goal state is reached or the container is empty (returns False if empty).
    """
    visited = []
    starting_node = [(problem.getStartState(), "", 0), []]
    state, route = starting_node[0][0], starting_node[1]
    while not problem.isGoalState(state):
        if state not in visited:
            visited.append(state)
            for successor in problem.getSuccessors(state):
                if successor[0] not in visited:
                    successor_route = route[:]
                    container.push((successor, successor_route + [successor[1]]))
        if container.isEmpty(): return False
        node = container.pop()
        state, route = node[0][0], node[1]
    return route

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # TODO Problem 1: DFS
    # For Depth First Search, a Stack is used as the container.
    container = util.Stack()
    return search(problem, container)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
    python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
    python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
    """
    # TODO Problem 2: BFS
    container = util.Queue()
    # For Breadth First Search, a Queue is used as the container.
    return search(problem, container)

def uniformCostSearch(problem):
    """Search the node of least total cost first.
     python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
     python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
     python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
     """
    # TODO Problem 3: UCS
    """
    For Uniform Cost Search, a Priority Queue is used as the container with a function that gets the cost of going
    through the route from the start state to the state of that node. Utilizes the second element of each successor.
    """
    def cost_function(node): return problem.getCostOfActions(node[1])
    container = util.PriorityQueueWithFunction(cost_function)
    return search(problem, container)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    """
    # TODO Problem 4: A*S
    """
    For A* search, a Priority Queue is used as the container with a function that gets the cost of going
    through the route from the start state to the state of that node, as UCS, 
    but also adds the Heuristic of the successor state.
    """
    def cost_function(node): return problem.getCostOfActions(node[1]) + heuristic(node[0][0], problem)
    container = util.PriorityQueueWithFunction(cost_function)
    return search(problem, container)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
