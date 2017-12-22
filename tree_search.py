import sys

import random

import gym
import numpy as np
from gym_paneldepon.env import register
from gym_paneldepon.state import TOP

register()

def tree_search_agent(env):
    root = env.unwrapped.get_root()
    best_score = -float("inf")
    best_action = None
    children = list(enumerate(root.get_children()))
    children.pop(0)  # Don't pass
    raise_stack = None
    if not any(panels & TOP for panels in root.colors):
        raise_stack = children.pop(0)
    random.shuffle(children)  # Eliminate bias on equally good moves
    if raise_stack:
        children.insert(0, raise_stack)  # But default to raising the stack
    for action, (child, score) in children:
        for grand_child, child_score in child.get_children():
            for _, grand_child_score in grand_child.get_children():
                total = 100 * score + 99 * child_score + 98 * grand_child_score
                if total > best_score:
                    best_action = action
                    best_score = total
    return best_action


if __name__ == "__main__":
    n = sys.argv[1]
    seed = random.randint(0, 1234567890)
    env = gym.make("PdPEndless4-v0")
    env.seed(seed)
    env.reset()

    print(seed)
    total_reward = 0
    with open("tree_search_{}.record".format(n), "w") as f:
        f.write(str(seed) + "\n")
        while True:
            # env.render()
            # print("Reward =", total_reward)
            action = tree_search_agent(env)
            f.write(str(action) + "\n")
            _, reward, _, _ = env.step(action)
            total_reward += reward
