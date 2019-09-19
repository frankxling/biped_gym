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
# env.reset()
count = 0
while True:
    # take a random action
    # if(count < 5):
    #     count = count +1
    # else:
    #     env.reset()
    #     print("resetting environment")
    #     time.sleep(3)
    #     count =0
    observation, reward, done, info = env.step(env.action_space.sample())
    # For actual implementation please do not sleep here, your observation will be outdated by the time the next loop comes around
    time.sleep(0.5)