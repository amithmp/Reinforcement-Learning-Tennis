import torch

print("Loading new constants")

TARGET_SCORE = 0.5

BUFFER_SIZE = int(1e6)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 3e-2                 # for soft update of target parameters
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 0      # decay rate for noise process
n_episodes=2000         
max_t=2000
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter


USE_LSTM = True         # Flag indicating whether to use LSTM or plan FC
NUM_LAYERS_LSTM = 2     # Number of layers in the LSTM if used
SEQUENCE_LEN = 1        # Length of sequence of inputs to LSTM. If more than 1, should process input accordingly


BATCH_SIZE = 256        # minibatch size
LR_ACTOR = 1e-3         # learning rate 
LR_CRITIC = 3e-3        # learning rate 
LR_DECAY = True
LR_DECAY_STEP = 1
LR_DECAY_GAMMA = 0.01
WEIGHT_DECAY = 0 
if USE_LSTM:
    OPTIMIZER = "RMSPROP"
else:
    OPTIMIZER = "ADAM"
UPDATE_EVERY = 1       # how often to update the network
NUM_EPOCHS = 2          # How many learning epochs/iterations per each update
NEURON_1 = 256          # Number of Neurons in the first (FC or LSTM) layer 
NEURON_2 = 64           # Number of neurons in the second (FC) layer
GRAD_CLIP = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

