import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Concatenate
from keras.optimizers import Adam
from keras import Input
from keras import Model

from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1

num_actions = env.action_space.shape[0]

# we use ddpg here because of space is continuous (physic model)

#building model
def build_model(num_actions):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(num_actions))
    actor.add(Activation('linear'))
    return actor

def build_critic_model(num_actions):
    action_input = Input(shape=(num_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return action_input, critic

model = build_model(num_actions)
critic_action_input, critic = build_critic_model(num_actions)

# reiforcement learning
## memory
# keras-rl provides us a class SequentialMemory that provides a fast and efficient data structure
# that we can store the agent's experiences
memory = SequentialMemory(limit=100000, window_length=1)

# random - process
random_process = OrnsteinUhlenbeckProcess(size=num_actions, theta=.15, mu=0., sigma=.3)

# agent
# With a model, memory, and policy defined, we’re now ready to create a deep Q network Agent and send that agent those objects. 
# Keras-RL provides an agent class called DDPG Agent that we can use for this, as shown in the following code:

# nb_steps_warmup: Determines how long we wait before we start doing experience replay, which if you recall, is when we actually start training the network. 
# This lets us build up enough experience to build a proper minibatch. 
# If you choose a value for this parameter that’s smaller than your batch size, Keras RL will sample with a replacement.

# target_model_update: The Q function is recursive and when the agent updates it’s network for Q(s,a) that update also impacts the prediction it will make for 
# Q(s’, a). This can make for a very unstable network. The way most deep Q network implementations address this limitation is by using a target network, which 
# is a copy of the deep Q network that isn’t trained, but rather replaced with a fresh copy every so often. The target_model_update parameter controls how often this happens.
ddpg = DDPGAgent(nb_actions=num_actions, actor=model, critic=critic, critic_action_input=critic_action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
ddpg.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

ddpg.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

ddpg.test(env, nb_episodes=5, visualize=True)
