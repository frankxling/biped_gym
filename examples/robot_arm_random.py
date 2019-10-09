#!/usr/bin/env python3
""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import time
import multiprocessing
import biped_gym
env = gym.make('RobotArm-v0')

env.reset()
count = 0
while True:
    for x in range(5):
        observation, reward, done, info = env.step(env.action_space.sample())
        time.sleep(0.5)
    time.sleep(1.0)
    print("-------------Resetting environment---------------")
    env.reset()
    print("-------------Reset finished----------------")
    time.sleep(2.0)

