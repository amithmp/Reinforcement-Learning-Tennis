import numpy as np
import random
import copy
from collections import namedtuple, deque

from constants import *

from ddpg_network import *

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, memory, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        self.memory = memory

        # Actor Network (w/ Target Network)
        if(USE_LSTM == True):
            self.actor_local = ActorLSTM(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
            self.actor_target = ActorLSTM(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
        else:
            self.actor_local = Actor(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
        if(OPTIMIZER == "ADAM"):
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        elif(OPTIMIZER == "RMSPROP"):
            self.actor_optimizer = optim.RMSprop(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
            
        # Critic Network (w/ Target Network)
        if(USE_LSTM == True):
            self.critic_local = CriticLSTM(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
            self.critic_target = CriticLSTM(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
        else:
            self.critic_local = Critic(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed, NEURON_1, NEURON_2).to(device)
        if(OPTIMIZER == "ADAM"):
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        elif(OPTIMIZER == "RMSPROP"):
            self.critic_optimizer = optim.RMSprop(self.critic_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)    
    
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
            
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
                
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, i_epoch):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        if(LR_DECAY):
            self.critic_scheduler.step(i_epoch)                
            self.actor_scheduler.step(i_epoch) 
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIP)
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), GRAD_CLIP)
        self.actor_optimizer.step()                       


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)   
        
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MultiAgent:
    def __init__(self, state_size, action_size, num_agents, random_seed, load_file=None):
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.ddpg_agents = [Agent(state_size=state_size, action_size=action_size, memory=self.memory, random_seed=random_seed) for _ in range(num_agents)]
        
        self.t_step = 0
        
        if load_file:
            file_suffix = load_file
            for agent_i, save_agent in enumerate(self.ddpg_agents):
                actor_path = file_suffix % ('actor', agent_i)
                critic_path = file_suffix % ('critic', agent_i)
                actor_file = torch.load(actor_path, map_location='cpu')
                critic_file = torch.load(critic_path, map_location='cpu')
                save_agent.actor_local.load_state_dict(actor_file)
                save_agent.actor_target.load_state_dict(actor_file)
                save_agent.critic_local.load_state_dict(critic_file)
                save_agent.critic_target.load_state_dict(critic_file)
            print('Loaded Actors at {} and Critics at {} '.format(actor_path,critic_path))
     
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
    
    def act(self, all_states, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(np.expand_dims(states, axis=0), add_noise) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if (self.t_step == 0) and (len(self.memory) > BATCH_SIZE):   
            for i_epoch in range(NUM_EPOCHS):
                for agent in self.ddpg_agents:
                    experiences = self.memory.sample()   
                    agent.learn(experiences, GAMMA, i_epoch)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)