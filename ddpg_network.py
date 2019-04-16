import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
  
    def __init__(self, state_size, action_size, seed, fc1_units=400,fc2_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units) 
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.normalizer_1 = nn.BatchNorm1d(state_size)
        self.normalizer_2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = self.normalizer_1(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class ActorLSTM(nn.Module):
    """Actor (Policy) Model."""
  
    def __init__(self, state_size, action_size, seed, lstm_units = 128, fc1_units = 32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorLSTM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.lstm_units = lstm_units
        self.fc1_units = fc1_units
        self.seed = torch.manual_seed(seed)        
        self.lstm = nn.LSTM(state_size, lstm_units, NUM_LAYERS_LSTM, batch_first=True)
        self.fc1 = nn.Linear(lstm_units, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        self.normalizer_1 = nn.BatchNorm1d(SEQUENCE_LEN)
        self.normalizer_2 = nn.BatchNorm1d(SEQUENCE_LEN)
        self.reset_parameters()
        
    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.normal(param)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""       
        state = state.reshape(-1, SEQUENCE_LEN, self.state_size).to(device)      
        h0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), self.lstm_units).to(device) 
        c0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), self.lstm_units).to(device)
        
        state = self.normalizer_1(state)
        # Forward propagate LSTM
        out, _ = self.lstm(state, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        #out = self.normalizer_2(out)
        out = F.relu(self.fc1(out[:, -1, :]))
        return F.tanh(self.fc2(out))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.normalizer_1 = nn.BatchNorm1d(state_size)
        self.normalizer_2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        xs = self.normalizer_1(state)
        xs = self.fcs1(xs)
        xs = F.relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class CriticLSTM(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, lstm_units=128, fc1_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticLSTM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.lstm_units = lstm_units
        self.fc1_units = fc1_units
        self.seed = torch.manual_seed(seed)        
        self.lstm = nn.LSTM(state_size, lstm_units, NUM_LAYERS_LSTM, batch_first=True)
        self.fc1 = nn.Linear(lstm_units + action_size , fc1_units)
        self.fc2 = nn.Linear(fc1_units , 1)
        self.normalizer_1 = nn.BatchNorm1d(SEQUENCE_LEN)
        self.normalizer_2 = nn.BatchNorm1d(SEQUENCE_LEN)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.normal(param)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        state = state.reshape(-1, SEQUENCE_LEN, self.state_size).to(device)
        action = action.reshape(BATCH_SIZE, SEQUENCE_LEN, self.action_size).to(device)
            
        h0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), self.lstm_units).to(device) 
        c0 = torch.zeros(NUM_LAYERS_LSTM, state.size(0), self.lstm_units).to(device)
        
        state = self.normalizer_1(state)
        # Forward propagate LSTM
        out, _ = self.lstm(state, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        out = F.relu(out)
        # Decode the hidden state of the last time step
        out = torch.cat((out, action), dim=2)
        #out = self.normalizer_2(out)
        out = F.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)