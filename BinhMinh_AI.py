# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat
# Modified by NGUYEN Binh Minh
# Improvement: Combining game theory, adding another layer of information in the canvas and design a more complex model.
###############################
# Team name to be displayed in the game 
TEAM_NAME = "Q-Learner + Game theory"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################
# Please put your global variables here

# Global variables
global model, exp_replay, input_tm1, action, score

# Function to create a numpy array representation of the maze

def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 2)
    canvas = np.zeros(im_size)
    (x,y) = player
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1
    (x_enemy, y_enemy) = opponent
    canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1
    #canvas[center_y,center_x,2] = 1
    canvas = np.expand_dims(canvas, axis=0)
    return canvas


class NLinearModels(nn.Module):
    def __init__(self, x_example, number_of_regressors=4):
        super(NLinearModels, self).__init__()
        in_features = x_example.reshape(-1).shape[0]
        
        self.linear1 = nn.Linear(in_features, 128)
        self.linear2 = nn.Linear(128, 16)
        self.out = nn.Linear(16, number_of_regressors)
    
    # Relu() is an activation function
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

    def load(self):
        self.load_state_dict(torch.load('BinhMinh.pt'))

    def save(self):
        torch.save(self.state_dict(), 'BinhMinh.pt')

    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model,exp_replay,input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    model = NLinearModels(input_tm1[0])
    model.load()
    
# Here we define the function for combining game theory.
# The first things we do is we program the AI of the opponent, so that we know exactly what will be its decision in a given situation
def distance(la, lb):
    ax,ay = la
    bx,by = lb
    return abs(bx - ax) + abs(by - ay)
    
def move(location, move):
    if move == MOVE_UP:
        return (location[0], location[1] + 1)
    if move == MOVE_DOWN:
        return (location[0], location[1] - 1)
    if move == MOVE_LEFT:
        return (location[0] - 1, location[1])
    if move == MOVE_RIGHT:
        return (location[0] + 1, location[1])

def turn_of_opponent(opponentLocation, piecesOfCheese):    
    closest_poc = (-1,-1)
    best_distance = -1
    for poc in piecesOfCheese:
        if distance(poc, opponentLocation) < best_distance or best_distance == -1:
            best_distance = distance(poc, opponentLocation)
            closest_poc = poc
    ax, ay = opponentLocation
    bx, by = closest_poc
    if bx > ax:
        return MOVE_RIGHT
    if bx < ax:
        return MOVE_LEFT
    if by > ay:
        return MOVE_UP
    if by < ay:
        return MOVE_DOWN
    pass

# We use a recursive function that goes through the trees of possible plays
# It takes as arguments a given situation, and return a best target piece of cheese for the player, such that aiming to grab this piece of cheese will eventually lead to a maximum score. It also returns the corresponding score
def best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):

    # First we should check how many pieces of cheese each player has to see if the play is over. It is the case if no pieces of cheese are left, or if playerScore or opponentScore is more than half the total number playerScore + opponentScore + piecesOfCheese
    totalPieces = len(piecesOfCheese) + playerScore + opponentScore
    if playerScore > totalPieces / 2 or opponentScore > totalPieces / 2 or len(piecesOfCheese) == 0:
        return (-1,-1), playerScore

    # If the play is not over, then the player can aim for any of the remaining pieces of cheese
    # So we will simulate the game to each of the pieces, which will then by recurrence test all
    # the possible trees.

    best_score_so_far = -1
    best_target_so_far = (-1,-1)
    for target in piecesOfCheese:
        end_state = simulate_game_until_target(
            target,playerLocation,opponentLocation,
            playerScore,opponentScore,piecesOfCheese.copy())
        _, score = best_target(*end_state)
        if score > best_score_so_far:
            best_score_so_far = score
            best_target_so_far = target

    return best_target_so_far, best_score_so_far

# Move the agent on the labyrinth using the function move and the different directions
# It suffices to move in the direction of the target. 
# You should only run function move once and you can't move diagonally.
# Without loss of generality, we can suppose it gets there moving vertically first then horizontally
def updatePlayerLocation(target,playerLocation):
    if playerLocation[1] != target[1]:
        if target[1] < playerLocation[1]:
            playerLocation = move(playerLocation, MOVE_DOWN)
        else:
            playerLocation = move(playerLocation, MOVE_UP)
    elif target[0] < playerLocation[0]:
        playerLocation = move(playerLocation, MOVE_LEFT)
    else:
        playerLocation = move(playerLocation, MOVE_RIGHT)
    return playerLocation

#CHECK IF EITHER/BOTH PLAYERS ARE ON THE SAME SQUARE OF A CHEESE. 
#If that is the case you have to remove the cheese from the piecesOfCheese list and 
#add points to the score. The players get 1 point if they are alone on the square with a cheese.
#If both players are in the same square and there is a cheese on the square each player gets 0.5 points.
def checkEatCheese(playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese):
    if playerLocation in piecesOfCheese and playerLocation == opponentLocation:
        playerScore = playerScore + 0.5
        opponentScore = opponentScore + 0.5
        piecesOfCheese.remove(playerLocation)
    else:
        if playerLocation in piecesOfCheese:
            playerScore = playerScore + 1
            piecesOfCheese.remove(playerLocation)
        if opponentLocation in piecesOfCheese:
            opponentScore = opponentScore + 1
            piecesOfCheese.remove(opponentLocation)
    return playerScore,opponentScore


#In this function we simulate what will happen until we reach the target
#You should use the two functions defined before
def simulate_game_until_target(target,playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese):
    
    #While the target cheese has not yet been eaten by either player
    #We simulate how the game will evolve until that happens    
    while target in piecesOfCheese:
        #Update playerLocation (position of your player) using updatePlayerLocation
        playerLocation = updatePlayerLocation(target,playerLocation)
        #Every time that we move the opponent also moves. update the position of the opponent using turn_of_opponent and move
        opponentLocation = move(opponentLocation, turn_of_opponent(opponentLocation, piecesOfCheese))
        #Finally use the function checkEatCheese to see if any of the players is in the same square of a cheese.
        playerScore, opponentScore = checkEatCheese(
            playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese)
    return playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese

# During our turn we continue going to the next target, unless the piece of cheese it originally contained has been taken
# In such case, we compute the new best target to go to
current_target = (-1,-1)

###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move


def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    global model,input_tm1, action, score, current_target
    
    if len(piecesOfCheese) >12: # The game theory method works well with a small number of cheeses, thus we set a threshold to use it.
        # Reinforcement learning model's actions
        input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
        input_tm1 = torch.FloatTensor(input_t)   
        output = model(input_tm1.unsqueeze(dim=0))
        action = torch.argmax(output[0]).item()
        score = playerScore
        return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]
    else:
        # Game theory method's actions
        if current_target not in piecesOfCheese:
            current_target, score = best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
                #print("My new target is " + str(current_target) + " and I will finish with " + str(score) + " pieces of cheese")        
        if current_target[1] > playerLocation[1]:
            return MOVE_UP
        if current_target[1] < playerLocation[1]:
            return MOVE_DOWN
        if current_target[0] > playerLocation[0]:
            return MOVE_RIGHT
        return MOVE_LEFT


def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
