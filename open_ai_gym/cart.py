import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras import Input
from keras import Model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

ENV_NAME = 'CartPole-v0'
num_actions = 2
state_size = 4

#building model
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model

model = build_model(state_size, num_actions)

# reiforcement learning
## memory
# keras-rl provides us a class SequentialMemory that provides a fast and efficient data structure
# that we can store the agent's experiences
memory = SequentialMemory(limit=50000, window_length=1)

# policy
# keras-rl provides a policy that we can use to balance exploration and exploitation.
# Here we’re saying that we want to start with a value of 1 for and go no smaller than 0.1, while testing if our random number
# is less than 0.05. We set the number of steps between 1 and .1 to 10,000 and Keras-RL handles the decay math for us.
policy = BoltzmannQPolicy()

# agent
# With a model, memory, and policy defined, we’re now ready to create a deep Q network Agent and send that agent those objects. 
# Keras-RL provides an agent class called DQNAgent that we can use for this, as shown in the following code:

# nb_steps_warmup: Determines how long we wait before we start doing experience replay, which if you recall, is when we actually start training the network. 
# This lets us build up enough experience to build a proper minibatch. 
# If you choose a value for this parameter that’s smaller than your batch size, Keras RL will sample with a replacement.

# target_model_update: The Q function is recursive and when the agent updates it’s network for Q(s,a) that update also impacts the prediction it will make for 
# Q(s’, a). This can make for a very unstable network. The way most deep Q network implementations address this limitation is by using a target network, which 
# is a copy of the deep Q network that isn’t trained, but rather replaced with a fresh copy every so often. The target_model_update parameter controls how often this happens.
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# callback
# Keras-RL provides several Keras-like callbacks that allow for convenient model checkpointing and logging.
def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

callbacks = build_callbacks(ENV_NAME)

env = gym.make(ENV_NAME)

dqn.fit(env, nb_steps=10000, visualize=True, verbose=2, callbacks=callbacks)

dqn.test(env, nb_episodes=5, visualize=True)