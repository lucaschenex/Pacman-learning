# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    returnScore=0.0
    
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanState().getPosition()
    Food = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodList = [util.manhattanDistance(newPos, f) for f in Food.asList()]
    foodList.sort()
    foodScore=foodList[0]
    
    ## can add shocktime into
    GhostPositions = [Ghost.getPosition() for Ghost in newGhostStates]
    GhostDistance = [util.manhattanDistance(newPos, g) for g in GhostPositions]
    if len(GhostPositions) ==0 : GhostScore=0
    else:
        GhostDistance.sort()
        if GhostDistance[0]==0: return -99
        else:
            GhostScore=2*-1.0/GhostDistance[0]
    if foodScore==0: 
        returnScore=2.0+GhostScore
    else: 
        returnScore=GhostScore+1.0/float(foodScore)
    return returnScore

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
        
        num_agent = gameState.getNumAgents()#
        totalDepth = num_agent*self.depth-1#
        actions = gameState.getLegalActions(0)#
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        
        state = [gameState.generateSuccessor(0, action) for action in actions]
        value = [self.minimax(num_agent, nxt, totalDepth) for nxt in state]
        Maxval = max(value)
        lMax = []
        for i in range(0, len(value)):
            if value[i] == Maxval:
                lMax.append(i)
        r = random.randint(0, len(lMax)-1)
    
        return actions[lMax[r]]
        
    def minimax(self, num_agent, gameState, depth):
        nameAgent = depth%num_agent
        legalActions = gameState.getLegalActions(nameAgent)
        nextStates = [gameState.generateSuccessor(nameAgent, action) for action in legalActions]
        
        if self.GameOver(gameState) or depth <= 0:
            return self.evaluationFunction(gameState)
        
        else:
            if nameAgent == 0:
                alpha = float('-inf')
            else:
                alpha = float('inf')
            for child in nextStates:
                if nameAgent == 0:
                    alpha = max(alpha, self.minimax(num_agent, child, depth-1))
                else:
                    alpha = min(alpha, self.minimax(num_agent, child, depth-1))
        return alpha

    def GameOver(self, gameState):
        if gameState.isWin() or gameState.isLose(): return True
        else: return False
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
            """
        "*** YOUR CODE HERE ***"
        
        num_agent = gameState.getNumAgents()#
        totalDepth = num_agent*self.depth-1#
        actions = gameState.getLegalActions(0)#
        alpha = float('-inf')
        beta = float('inf')
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        value = []
        state = [gameState.generateSuccessor(0, action) for action in actions]
        for nxt in state:
                score = self.alphabeta(num_agent, nxt, totalDepth, alpha, beta)
                value.append(score)
                alpha = max(alpha, score)
        t = max(value)
        lMax = []
        for i in range(0, len(value)):
            if value[i] == t:
                lMax.append(i)
        r = random.randint(0, len(lMax)-1)
        
        return actions[lMax[r]]
            
    def alphabeta(self, num_agent, gameState, depth, alpha, beta):
        nameAgent = depth%num_agent
        legalActions = gameState.getLegalActions(nameAgent)
        nextStates = [gameState.generateSuccessor(nameAgent, action) for action in legalActions]
            
        if self.GameOver(gameState) or depth <= 0:
            return self.evaluationFunction(gameState)
            
        for child in nextStates:
            if nameAgent == 0:
                alpha = max(alpha, self.alphabeta(num_agent, child, depth-1, alpha, beta))
                if beta <= alpha:
                    break
            else:
                beta = min(beta, self.alphabeta(num_agent, child, depth-1, alpha, beta))
                if beta <= alpha:
                    break
            
        if nameAgent == 0:
            return alpha
        return beta
            
    def GameOver(self, gameState):
        if gameState.isWin() or gameState.isLose(): return True
        else: return False


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
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()#
        totalDepth = num_agent*self.depth-1#
        actions = gameState.getLegalActions(0)#
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
    
        state = [gameState.generateSuccessor(0, action) for action in actions]
        value = [self.expectiminimax(num_agent, nxt, totalDepth) for nxt in state]
        Maxval = max(value)
        lMax = []
        for i in range(0, len(value)):
            if value[i] == Maxval:
                lMax.append(i)
        return actions[random.choice(lMax)]
        
    
    def expectiminimax(self, num_agent, gameState, depth):
        nameAgent = depth%num_agent
        legalActions = gameState.getLegalActions(nameAgent)
        nextStates = [gameState.generateSuccessor(nameAgent, action) for action in legalActions]
            
        if self.GameOver(gameState) or depth <= 0:
            return self.evaluationFunction(gameState)
        
        else:
            if nameAgent == 0:
                alpha = float('-inf')
                for child in nextStates:
                    alpha = max(alpha, self.expectiminimax(num_agent, child, depth-1))
                return alpha
            else:
                alpha = 0
                for child in nextStates:
                    alpha = alpha + (self.expectiminimax(num_agent, child, depth-1)/len(nextStates))
                return alpha
        
    def GameOver(self, gameState):
        if gameState.isWin() or gameState.isLose(): return True
        else: return False
    

def foodReward(currentGameState):
    curPos = currentGameState.getPacmanState().getPosition()
    Food = currentGameState.getFood()
    Capsules = currentGameState.getCapsules()
    
    # 1 it is good to minimize the distance of the farest food
    # 2 it is good to minimize the total amount of food
    # 3 it is good to minimize the distance of the nearest food
    # 4 it is good to go find the pellete and eat it if it's near by
    
    # the food reward should not go beyond the fear of the ghost
    
    foodDistances = [util.manhattanDistance(curPos, food) for food in Food]

    numFood = 0
    for g in Food:
        for i in g:
            if i:
                numFood = numFood + 1
    
    # extreme cases:
    if numFood == 0:
        return 1.1                                     # function exit
    if len(Capsules)>0 :
        CapsulesDistances = [util.manhattanDistance(curPos, capsl) for capsl in Capsules]
        CapsulesDistances.sort()
    if len(Capsules) == 0 or len(Capsules) == 1:
        return 1.1
    
    foodDistances.sort()
    farestFood = max(foodDistances)
    nearestFood = foodDistances[0]
    if nearestFood == 0:
        return 1.1
    else:
        returnValue = 1.0/float(nearestFood)

    returnValue = -numFood

#    returnValue = returnValue*100%20
    return returnValue + 5.0/3.0 * float(numFood)                                    # function exit

def ghostPenalty(currentGameState):
    curPos = currentGameState.getPacmanState().getPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghost.scaredTimer for ghost in GhostStates]
    
    # 1 a ghost that is scared is not something to worry about
    # 2 it is bad to have the nearest ghost too close by
    # 3 it is good to keep the total distance of the two ghost high enough
    
    activeGhosts = [ghost for ghost in GhostStates if ghost.scaredTimer <= 0]
    ghostDistances = [util.manhattanDistance(curPos, ghost.getPosition()) for ghost in activeGhosts]
    ghostDistances.sort()
    if len(ghostDistances) == 0:                        # function exit
        return 0
    else:
        nearestGhost = min(ghostDistances)
    totalGhostDistance = sum(ghostDistances)
    returnValue = 0
            
    # extreme cases:
    if nearestGhost == 0:   return -99                  # function exit
    else:
        returnValue = 1.0 - 1.0/float(nearestGhost)
        return returnValue                              # function exit

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
                    Basically, I was just bringing scaredtimer into account.

    """
    "*** YOUR CODE HERE ***"
    returnScore=0.0
    
    newPos = currentGameState.getPacmanState().getPosition()
    Food = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodList = [util.manhattanDistance(newPos, f) for f in Food.asList()]
    foodScore = 0
    
    for f in foodList:
        foodScore += 1.0/float(f)
    
    ## can add shocktime into
    GhostPositions = [Ghost.getPosition() for Ghost in newGhostStates]
    GhostDistance = [util.manhattanDistance(newPos, g) for g in GhostPositions]
    GhostDistance.sort()
    
    GhostScore = 0
    if min(GhostDistance) == 0:
        GhostScore == 100000000
    else:
        for g in GhostDistance:
            if g < 3 and g != 0:
                GhostScore + 1.0/g
    
    scaretimeSum = sum(newScaredTimes)

    return currentGameState.getScore() + foodScore - 28*GhostScore + 1.2 * scaretimeSum



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

#