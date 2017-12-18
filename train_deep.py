from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import deque
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_paneldepon.env import register
from gym_paneldepon.util import print_up

from util import bias_variable, conv2d, summarize_scalar, variable_summaries, vh_log, weight_variable

register()


FLAGS = None


class Agent(object):
    BATCH_SIZE = 1
    HISTORY_SIZE = 4
    KERNEL_SIZE = 5
    NUM_FEATURES = 20
    FC_1_SIZE = 200
    FC_2_SIZE = 200
    REPLAY_SIZE = 1000
    GAMMA = 0.99

    def __init__(self, session):
        self.session = session
        self.env = gym.make("PdPEndless4-v0")
        self.make_graph()
        self.make_summaries()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(tf.get_default_graph())

        self.replay_table = deque(maxlen=self.REPLAY_SIZE)

    def make_graph(self):
        self.make_input_graph()
        if FLAGS.use_convolution:
            self.make_convolution_graph()
        else:
            self.box_activations = self.box_inputs
            self.NUM_FEATURES = self.box_shape[-1]
        self.make_fc_1_graph()
        self.make_fc_2_graph()
        self.make_output_graph()
        self.make_loss_graph()
        self.make_train_graph()

    def make_summaries(self):
        if FLAGS.use_convolution:
            variable_summaries(self.W_conv, "W_conv")
            variable_summaries(self.b_conv, "b_conv")
        variable_summaries(self.W_fc_1, "W_fc_1")
        variable_summaries(self.b_fc_1, "b_fc_1")
        variable_summaries(self.W_fc_2, "W_fc_2")
        variable_summaries(self.b_fc_2, "b_fc_2")
        variable_summaries(self.W_output, "W_output")
        variable_summaries(self.b_output, "b_output")

        tf.summary.histogram("Q", self.output)

    def make_input_graph(self):
        chain_space, box_space = self.env.observation_space.spaces
        with tf.name_scope("input"):
            self.chain_inputs = []
            self.box_inputs = []
            for i in range(self.HISTORY_SIZE):
                with tf.name_scope("frame_{}".format(i)):
                    self.chain_inputs.append(tf.placeholder(tf.float32, [self.BATCH_SIZE, 1], name="chain"))
                    self.box_inputs.append(
                        tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(box_space.shape), name="box")
                    )
        self.box_shape = box_space.shape
        self.n_inputs = self.HISTORY_SIZE * (np.prod(self.box_shape) + 1)

    def make_convolution_graph(self):
        with tf.name_scope("convolution"):
            self.W_conv = weight_variable([self.KERNEL_SIZE, self.KERNEL_SIZE, self.box_shape[0], self.NUM_FEATURES], name="W")
            self.b_conv = bias_variable([self.NUM_FEATURES], name="b")
            self.box_activations = []
            for box in self.box_inputs:
                z = conv2d(box, self.W_conv) + self.b_conv
                self.box_activations.append(tf.sigmoid(z))

    def make_fc_1_graph(self):
        conv_layer_size = self.NUM_FEATURES * self.box_shape[1] * self.box_shape[2]
        with tf.name_scope("flatten"):
            flat_input = self.chain_inputs[:]
            for activation in self.box_activations:
                flat_input.append(tf.reshape(activation, [-1, conv_layer_size]))
            flat_input = tf.concat(flat_input, 1)
        n_flat = self.HISTORY_SIZE * (1 + conv_layer_size)

        with tf.name_scope("fully_connected_1"):
            self.W_fc_1 = weight_variable([n_flat, self.FC_1_SIZE], name="W")
            self.b_fc_1 = bias_variable([self.FC_1_SIZE], name="b")
            z = tf.matmul(flat_input, self.W_fc_1) + self.b_fc_1
            self.fc_1_activation = tf.sigmoid(z)

    def make_fc_2_graph(self):
        with tf.name_scope("fully_connected_2"):
            self.W_fc_2 = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2 = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2) + self.b_fc_2
            self.fc_2_activation = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_outputs = self.env.action_space.n
        with tf.name_scope("output"):
            self.W_output = weight_variable([self.FC_2_SIZE, self.n_outputs], name="W")
            self.b_output = bias_variable([self.n_outputs], name="b")
            z = tf.matmul(self.fc_2_activation, self.W_output) + self.b_output
            self.output = z
            self.action = tf.argmax(self.output, 1)

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_outputs], name="target")
            with tf.name_scope("error"):
                self.error = tf.reduce_sum(tf.square(self.output - self.target))
            with tf.name_scope("L2-norm"):
                self.L2_norm = sum(tf.reduce_sum(tf.square(variable)) for variable in self.variables)
            self.loss = self.error + self.L2_norm * 1e-7

    def make_train_graph(self):
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, states):
        feed_dict = {}
        for i, state in enumerate(states):
            chain, box = state
            feed_dict[self.chain_inputs[i]] = [[chain]]
            feed_dict[self.box_inputs[i]] = [box]
        return feed_dict

    def get_action(self, states):
        return self.session.run(self.action, feed_dict=self.get_feed_dict(states))

    def record_experience(self, experience):
        self.replay_table.append(experience)

    def learn(self, index):
        states, action, reward, new_state = self.replay_table[index]
        new_states = states[1:] + [new_state]
        Q_base = self.session.run(self.output, feed_dict=self.get_feed_dict(states))  # noqa: N806
        Q = self.session.run(self.output, feed_dict=self.get_feed_dict(new_states))  # noqa: N806
        Q_target = Q_base  # noqa: N806
        Q_target[0, action[0]] = reward + self.GAMMA * np.max(Q)
        feed_dict = self.get_feed_dict(states)
        feed_dict[self.target] = Q_target
        self.session.run(self.train_step, feed_dict=feed_dict)

    def learn_from_memory(self):
        history_size = len(self.replay_table)
        if history_size > 1:
            self.learn(np.random.randint(0, history_size - 1))

    def learn_from_last(self):
        self.learn(-1)

    def render_in_place(self):
        self.env.render()
        print_up(5)

    def render_ansi(self):
        sio = self.env.render("ansi")
        print_up(5, outfile=sio)
        return sio

    @property
    def variable_names(self):
        if FLAGS.use_convolution:
            names = ["W_conv", "b_conv"]
        else:
            names = []
        return names + [
            "W_fc_1", "b_fc_1",
            "W_fc_2", "b_fc_2",
            "W_output", "b_output",
        ]

    @property
    def variables(self):
        return [getattr(self, name) for name in self.variable_names]

    def dump(self):
        outputs_dir = os.getenv("VH_OUTPUTS_DIR", "/tmp/tensorflow/gym_paneldepon/outputs")
        if not os.path.isdir(outputs_dir):
            os.makedirs(outputs_dir)
        arrays = self.session.run(self.variables)
        for arr, name in zip(arrays, self.variable_names):
            arr = arr.flatten()
            filename = os.path.join(outputs_dir, "{}.csv".format(name))
            np.savetxt(filename, arr, delimiter=",")
        print("Saved parameters to {}".format(outputs_dir))

    def load(self, params_dir):
        for variable, name in zip(self.variables, self.variable_names):
            filename = os.path.join(params_dir, "{}.csv".format(name))
            arr = np.loadtxt(filename, delimiter=",")
            arr = arr.reshape(variable.shape)
            self.session.run(variable.assign(arr))
        print("Loaded parameters from {}".format(params_dir))


def main(*args, **kwargs):
    with tf.Session() as session:
        agent = Agent(session)
        merged = tf.summary.merge_all()
        states = deque(maxlen=agent.HISTORY_SIZE)
        session.run(tf.global_variables_initializer())
        if FLAGS.params_dir:
            agent.load(FLAGS.params_dir)
        for i in range(FLAGS.num_episodes):
            exploration = FLAGS.exploration / (0.1 * i + 2)
            vh_log({"exploration": exploration}, i)
            total_reward = 0
            state = agent.env.reset()
            # Fully raise the panel stack
            for k in range(agent.HISTORY_SIZE):
                state, _, _, _ = agent.env.step(1)
                states.append(state)
            frames = []
            for j in range(FLAGS.num_steps):
                action = agent.get_action(list(states))
                if np.random.rand(1) < exploration:
                    action[0] = agent.env.action_space.sample()
                new_state, reward, _, _ = agent.env.step(action[0])
                agent.record_experience((list(states), action, reward, new_state))
                agent.learn_from_memory()
                agent.learn_from_last()
                if FLAGS.do_render:
                    # We only render 10% from the start of the episode so as not to clog the console.
                    frames.append(agent.render_ansi().getvalue())
                    if j % 10 == 0:
                        print(frames.pop(0), end="")
                total_reward += reward
                state = new_state
                states.append(state)
            if FLAGS.do_render:
                for _ in range(5):
                    print()
            vh_log({"reward": total_reward}, i)
            summarize_scalar(agent.writer, "Reward", total_reward, i)
            summary = session.run(merged, feed_dict=agent.get_feed_dict(list(states)))
            agent.writer.add_summary(summary, i)
        agent.dump()
        agent.writer.close()


def main_with_render(*args, **kwargs):
    try:
        print("\033[?25l")
        main(*args, **kwargs)
    finally:
        print("\033[?25h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to run the trainer")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps per episode")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorflow/gym_paneldepon/logs/rl_with_summaries",
                        help="Summaries log directory")
    parser.add_argument("--no_render", action="store_true",
                        help="Don't render visuals for episodes")
    parser.add_argument("--params_dir", type=str, default=None,
                        help="Parameters directory for initial values")
    parser.add_argument("--exploration", type=float, default=1.0,
                        help="Initial level of exploration for training")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.do_render = not FLAGS.no_render
    main_fun = main_with_render if FLAGS.do_render else main
    FLAGS.use_convolution = False
    tf.app.run(main=main_fun, argv=[sys.argv[0]] + unparsed)
