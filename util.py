from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import json

import gym
from gym_paneldepon.env import register
import numpy as np
import tensorflow as tf


def weight_variable(shape, name, stddev=None):
    """
    Create a weight variable with appropriate initialization.
    Defaults to Xavier initialization.
    """
    if stddev is None:
        stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(stddev=stddev, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, value=0.0):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(input, kernel):
    """Returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries-{}'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def summarize_scalar(writer, tag, value, step):
    """Add a custom summary outside of the main graph."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def vh_log(data, step):
    """Log data for valohai."""
    data["step"] = step
    print(json.dumps(data))


def parse_record(lines, num_frames=3):
    """
    Parses a seed + action record into a trainable sequence
    """
    lines = list(map(int, lines))
    seed = lines[0]
    actions = lines[1:]

    env = gym.make("PdPEndless4-v0")
    env.seed(seed)
    env.reset()
    states = []
    rewards = []
    for action in actions:
        state, reward, _, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

    values = []
    value = 0
    gamma = 0.95
    for reward in reversed(rewards):
        value = reward + gamma * value
        values.insert(0, value)

    result = []
    frames = deque(maxlen=num_frames)
    for state, action, value in list(zip(states, actions, values))[:-100]:
        frames.append(state)
        action_one_hot = np.zeros(env.action_space.n)
        action_one_hot[action] = 1
        if len(frames) == num_frames:
            yield (list(frames), action_one_hot, value)
