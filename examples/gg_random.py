""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import time
import multiprocessing
import biped_gym
env = gym.make('Biped-v0')
#env = gym.make('MARAOrient-v0')
#env = gym.make('MARACollision-v0')
#env = gym.make('MARACollisionOrient-v0')
#env = gym.make('MARACollisionOrientRandomTarget-v0')


# time.sleep(6)
while True:
    # take a random action
    observation, reward, done, info = env.step(env.action_space.sample())
    time.sleep(1)