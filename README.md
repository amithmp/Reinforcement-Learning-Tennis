# Reinforcement-Tennis
Create a Reinforcement agent to play tennis

This project aims to create an agent that plays tennis rather that control tennis racket. Multi-agent reinforcement technique as described in the paper https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf is implemented to create the agent. 

![Robot Arm](https://github.com/amithmp/Reinforcement-Learning-Tennis/blob/master/tennis.gif)


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Getting Started

1. Download the environment (after installing the dependancies as per next section) from one of the links below.  You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 


# Solution 

Agent is created using the techniques mentioned in the above paper, which is suitable for multi agent environments. This agent could solve the environment in 446 (average score for 100-episodes in 546th episode) episodes as below.

![Robot Arm](https://github.com/amithmp/Reinforcement-Learning-Tennis/blob/master/result_chart.png)

## Algorithm and Hyperparameters

This agent uses 2-layer LSTM network for both actor and critic. LSTM is chosen with the intuition that ideal action of the agent depends on the previous action, state etc and LSTM is good to represent temporal sequences. Number of neurons is set to 256 to learn complex representation and relationship among dimensions of the state. 

**Learning Method**: Both actor and critic are set to learn (i.e. network is updated) every 2 timesteps. Each learning step involves two epochs to stabilize the network.

**Learning Rate**: Learning rate is chosen to 1e-3 for the actor and 3e-3 after multiple experiments. Further, learn rate scheduler is used wherein the learning rate decays by a factor of 0.01 at each epoch to learn with smaller learning rates and stabilize the network.

**Optimizer**: RMSPROP is used after trying ADAM initially. RMSPROP is found to be suitable for RNNs in many cases.

**Tau(parameter that controls soft update of target networks)**: Tau is set to 3e-2, slightly higher than the value used for other environments such as continuous control of robotic arm. This is episodic task and also the number of time steps to complete an episode is relatively smaller , therefore smaller Tau value is required to ensure target networks are adequately updated. 

# Training and Using the agent

Run the cells from 1 to 8 in Tennis.ipynb to train the RL agent. You can modify various hyperparameters in constants.py to finetune the algorithm. Program will terminate after attaining the target score (set to 0.5 in constants.py) or after reaching maximum number of episodes. Trained agent (weights of actor and critic networks for both the agents) are saved in files names starting from "checkpoint...".

Run the cell number 10 to load the stored agent and see the tennis playing agents in action (visualization not supported). You can expect most of the games to end with score of more than 2.5. If you running this cell (10) to use the agent independently, ensure to run the cells 1 to 5 to set the environment.
