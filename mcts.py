from __future__ import division

from collections import deque
import json
import pickle
import random
import sys

import numpy as np
import gym
from gym_paneldepon.env import register
from gym_paneldepon.state import State, ACTIONS

from util import GAMMA

EXPLORATION = 8
PLAYOUT_LENGTH = 50


class RandomAgent(object):
    HISTORY_SIZE = 1

    def __init__(self):
        self.env = gym.make("PdPEndless4-v0")

    def get_action(self, frames):
        return self.env.action_space.sample()


class Node(object):
    def __init__(self, state, reward=0, action=None):
        self.state = state
        self.reward = reward
        self.action = action
        self.score = 0
        self.visits = 1
        self.children = []

    def expand(self):
        for action, (child, reward) in enumerate(self.state.get_children()):
            self.children.append(Node(child, reward, action))

    def choose(self, exploration):
        best_value = float("-inf")
        best_child = None
        random.shuffle(self.children)
        for child in self.children:
            value = child.value / child.visits
            value += exploration * np.sqrt(np.log(self.visits) / child.visits)
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def render(self):
        self.children.sort(key=lambda child: child.action)
        self.state.render()
        print("{} / {} = {}".format(self.score, self.visits, self.score / self.visits))
        for child in self.children:
            print("  {} / {} = {}".format(child.value, child.visits, child.value / child.visits))

    @property
    def value(self):
        return self.score + self.reward

    @property
    def confidence(self):
        best_child = self.choose(0)
        return best_child.visits / self.visits


def playout(agent, frames, state):
    rewards = []
    for _ in range(PLAYOUT_LENGTH):
        action = agent.get_action(frames)
        reward = state.step(ACTIONS[action])
        reward = min(reward, agent.env.unwrapped.max_chain)
        rewards.append(reward)

    score = 0
    for reward in reversed(rewards):
        score = reward + GAMMA * score
    return score


def mc_iterate(agent, node, frames, exploration=EXPLORATION):
    frames = deque(frames, maxlen=frames.maxlen)
    path = [node]
    while node.children:
        node = node.choose(exploration)
        path.append(node)
        frames.append(node.state.encode())
    node.expand()
    score = playout(agent, frames, node.state.clone())

    for node in reversed(path):
        node.visits += 1
        node.score += score
        score = node.reward + score * GAMMA


def mcts(agent):
    """
    Does a Monte Carlo tree search using an agent for rollout policy
    """
    seed = random.randint(0, 1234567890)
    env = gym.make("PdPEndless4-v0")
    env.seed(seed)
    frame = env.reset()
    frames = deque([frame] * agent.HISTORY_SIZE, maxlen=agent.HISTORY_SIZE)
    root = Node(env.unwrapped.get_root())

    exploration = int(sys.argv[1])
    print("Exploration", exploration)

    with open("mcts_exploration_{}.record".format(exploration), "w") as f:
        f.write(str(seed) + "\n")
        while True:
            for _ in range(500):
                mc_iterate(agent, root, frames, exploration)
            i = 0
            while root.confidence < 0.2 and i < 5:
                i += 1
                print("Gaining more confidence... {} %".format(100 * root.confidence))
                for _ in range(100):
                    mc_iterate(agent, root, frames, exploration)
            root.render()
            action = root.choose(0).action
            if random.random() < 0.05:
                print("Playing a random move")
                action = env.action_space.sample()
            f.write(str(action) + "\n")
            env.step(action)
            if action == 1:
                root = Node(env.unwrapped.get_root())
            else:
                # RNG not triggered we can reuse the MC tree
                root = root.children[action]

            frame = root.state.encode()
            frames.append(frame)

if __name__ == "__main__":
    register()
    mcts(RandomAgent())
