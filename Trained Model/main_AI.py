### Author: Carlos Lassance, Myriam Bontonou, Nicolas Farrugia
### Modified by NGUYEN Binh Minh
### Improvement: Combining Epsilon Greedy Strategy, Q selections and plots.

### The goal of this file is to use reinforcement learning to train an agent to play PyRat.
### We perform Q-Learning using a 3-layer network to predict the Q-values associated with each of the four possible movements. 
### This network is implemented with pytorch
### Use Experience Replay: while playing, the agent will 'remember' the moves he performs, and will subsequently train itself to 
### predict what should be the next move, depending on how much reward is associated with the moves.

### Usage : python main_AI.py



import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from AIs import manh
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

### The game_pyrat_simulation.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent.
import game_pyrat_simulation

### The Rlearning.py file describes the reinforcement learning procedure, including Q-learning, Experience replay, and a pytorch model to learn the Q-function.
### SGD is used to approximate the Q-function.
import Rlearning


### This set of parameters can be changed in your experiments.
### Definitions :
### - An iteration of training is called an Epoch. It correspond to a full play of a PyRat game. 
### - An experience is a set of  vectors < s, a, r, s’ > describing the consequence of being in state s, doing action a, receiving reward r, and ending up in state s'.
###   Look at the file rl.py to see how the experience replay buffer is implemented. 
### - A batch is a set of experiences we use for training during one epoch. We draw batches from the experience replay buffer.


epoch = 10000# Total number of epochs that will be done

max_memory = 1000  # Maximum number of experiences we are storing
number_of_batches = 8  # Number of batches per epoch
batch_size = 32  # Number of experiences we use for training per batch
width = 21  # Size of the playing field
height = 15  # Size of the playing field
cheeses = 40  # Number of cheeses in the game
opponent = manh  # AI used for the opponent

### If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters.
load = False
save = True

env = game_pyrat_simulation.PyRat()
exp_replay = Rlearning.ExperienceReplay(max_memory=max_memory)
model = Rlearning.NLinearModels(env.observe()[0])

if load:
    model.load()

def predict_action(explore_start, explore_stop, decay_rate, decay_step_, state, model):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step_)
    
    if explore_probability > np.random.rand():
        # Make a random action (exploration)
        action = random.choice([0,1,2,3])

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Q values state
        with torch.no_grad():
            q = model(state.unsqueeze(dim=0))
            action = torch.argmax(q[0]).item()
                
    return action 

def play(model, epochs, train=True):
    global win_rate, total_loss
    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    loss = 0.
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0
    t = 0
    win_rate = []
    total_loss = []
    
    explore_probability = 0.99
    # Decay step deciding how to choose an action for a next state
    # Exploration parameters
    #epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.0001
    decay_step = 0

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False
        t = t+1
        model.eval()
        input_t = torch.FloatTensor(env.observe())
        while not game_over:
            decay_step += 1
            input_tm1 = input_t        

            # with torch.no_grad():
            #     q = model(input_tm1.unsqueeze(dim=0))
            #     action = torch.argmax(q[0]).item()
        
            # Applying Epsilon Greedy Algorithm. 
            # At the beginning epochs, explore_probability is pretty high (due to the decay rate and step), thus the network try to explore new actions, which help the network learn faster.
            # The decay step increases along with the epochs trained, which leads to the decrease in explore_probability.
            # Therefore after some ecpochs, the network gets actions from Q-network (exploitation) more regularly.
            action = predict_action(max_epsilon, min_epsilon, decay, decay_step, input_tm1, model)
            
            # Apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            input_t = torch.FloatTensor(input_t)

            # Statistics
            if game_over:
                t = 0
                steps += env.round
                if env.score > env.enemy_score:
                    win_cnt += 1
                elif env.score == env.enemy_score:
                    draw_cnt += 1
                else:
                    lose_cnt += 1
                cheese = env.score
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)    

        win_hist.append(win_cnt)  # Statistics
        cheeses.append(cheese)  # Statistics

        if train:
            #model.cuda()
            model.train()
            local_loss = 0
            for _ in range(number_of_batches):                
                inputs, targets = exp_replay.get_batch(model,batch_size=batch_size)
                batch_loss = Rlearning.train_on_batch(model, inputs, targets, criterion, optimizer)
                local_loss += batch_loss
            loss += local_loss

        if (e+1) % 100 == 0:  # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            string = "Epoch {:03d}/{:03d} | Loss {:.4f} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}".format(
                        e,epochs, loss, cheese_np.sum(), 
                        cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt, 
                        win_cnt-last_W, draw_cnt-last_D, lose_cnt-last_L, steps/100)
            print(string)
            total_loss.append(loss)
            loss = 0.
            steps = 0.
            win_rate.append((win_cnt-last_W)/100) # Save winning history to plot
            total_loss.append(loss) # Save loss history to plot
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt  
             

print("Training")
play(model, epoch, True)
if save:
    model.save()
print("Training done")

# Save winning history in a file
f = open('win_rate_train.pckl', 'wb')
pickle.dump(win_rate, f)
f.close()
# Save loss history in a file
f = open('loss_train.pckl', 'wb')
pickle.dump(total_loss, f)
f.close()

# Plot the winning rate and loss associate with the number of epochs
plt.figure (1)
plt.plot(win_rate)
plt.title("winning rate")
plt.figure (2)
plt.plot(total_loss)
plt.title("Loss")
plt.show() 

print("Testing")
play(model, 1000, False)
print("Testing done")

# Plot testing result
plt.figure (3)
plt.plot(win_rate)  
plt.show()
