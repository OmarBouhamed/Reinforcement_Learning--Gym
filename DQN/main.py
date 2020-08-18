import os
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

from DQNAgent import *


# environment
env_id = "CartPole-v0" 
 # "MountainCar-v0"
env = gym.make(env_id)

###############################################################################
#Fixing seeds to have similar results
seed = 33

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)
############################ MAIN #############################################

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 64
target_update = 100
epsilon_decay = 1 / 2000

#Build our smart agent
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)


#train model
agent.train(num_frames)


#test model
frames = agent.test()





















