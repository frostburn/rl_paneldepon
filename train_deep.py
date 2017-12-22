from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict, deque
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_paneldepon.env import register
from gym_paneldepon.util import print_up

from util import bias_variable, conv2d, summarize_scalar, variable_summaries, vh_log, weight_variable, parse_record

register()


FLAGS = None


class Agent(object):
    BATCH_SIZE = 10
    HISTORY_SIZE = 3
    KERNEL_SIZE = 5
    NUM_FEATURES = 20
    FC_1_SIZE = 200
    FC_2_SIZE = 200

    def __init__(self, session):
        self.session = session
        self.env = gym.make("PdPEndless4-v0")
        self.make_graph()
        self.make_summaries()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(tf.get_default_graph())

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
        for variable, name in zip(self.variables, self.variable_names):
            variable_summaries(variable, name)
        tf.summary.histogram("policy_head", self.policy_head)
        tf.summary.histogram("value_head", self.value_head)

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
        # self.n_inputs = self.HISTORY_SIZE * (np.prod(self.box_shape) + 1)

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
        with tf.name_scope("fully_connected_2p"):
            self.W_fc_2_policy = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2_policy = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2_policy) + self.b_fc_2_policy
            self.fc_2_activation_policy = tf.sigmoid(z)

        with tf.name_scope("fully_connected_2v"):
            self.W_fc_2_value = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2_value = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2_value) + self.b_fc_2_value
            self.fc_2_activation_value = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        with tf.name_scope("policy"):
            self.W_policy = weight_variable([self.FC_2_SIZE, self.n_actions], name="W")
            self.b_policy = bias_variable([self.n_actions], name="b")
            self.policy_head = tf.matmul(self.fc_2_activation_policy, self.W_policy) + self.b_policy

        with tf.name_scope("value"):
            self.W_value = weight_variable([self.FC_2_SIZE, 1], name="W")
            self.b_value = bias_variable([1], name="b")
            self.value_head = tf.matmul(self.fc_2_activation_value, self.W_value) + self.b_value

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.policy_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_actions], name="policy_target")
            self.value_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, 1], name="value_target")
            with tf.name_scope("error"):
                self.loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.policy_target, logits=self.policy_head)
                self.loss_mse = tf.reduce_mean(tf.squared_difference(self.value_head, self.value_target))
            with tf.name_scope("regularization"):
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
                reg_variables = tf.trainable_variables()
                self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            self.loss = self.loss_xent + self.loss_mse + self.reg_term

    def make_train_graph(self):
        with tf.name_scope("train"):
            # self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9, use_nesterov=True)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, experiences):
        feed_dict = defaultdict(list)
        for states, actions, value in experiences:
            for chain_input, box_input, state in zip(self.chain_inputs, self.box_inputs, states):
                chain, box = state
                feed_dict[chain_input].append([chain])
                feed_dict[box_input].append(box)
            feed_dict[self.policy_target].append(actions)
            feed_dict[self.value_target].append([value])
        return dict(feed_dict)

    def get_policy_action(self, states):
        experiences = [(states, [0] * self.n_actions, 0)] * self.BATCH_SIZE
        feed_dict = self.get_feed_dict(experiences)
        actions = self.session.run(self.policy_head, feed_dict=feed_dict)[0]
        print(actions)
        return np.argmax(actions)

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
            "W_fc_2_policy", "b_fc_2_policy",
            "W_fc_2_value", "b_fc_2_value",
            "W_policy", "b_policy",
            "W_value", "b_value",
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
        session.run(tf.global_variables_initializer())
        if FLAGS.params_dir:
            agent.load(FLAGS.params_dir)
        i = 0
        while True:
            for n in range(1, 9):
                with open("tree_search_records/tree_search_{}.record".format(n)) as f:
                    g = parse_record(f.readlines(), num_frames=Agent.HISTORY_SIZE)
                experiences = deque(maxlen=agent.BATCH_SIZE)
                try:
                    while True:
                        for _ in range(agent.BATCH_SIZE):
                            experiences.append(next(g))
                        feed_dict = agent.get_feed_dict(experiences)
                        session.run(agent.train_step, feed_dict=feed_dict)
                        if i % 10 == 0:
                            summary = session.run(merged, feed_dict=feed_dict)
                            agent.writer.add_summary(summary, i)
                        i += 1
                except StopIteration:
                    pass

                total_reward = 0
                state = agent.env.reset()
                states = deque([state] * agent.HISTORY_SIZE, maxlen=agent.HISTORY_SIZE)
                for j in range(100):
                    state, reward, _, _ = agent.env.step(agent.get_policy_action(list(states)))
                    states.append(state)
                    total_reward += reward
                    agent.env.render()
                print(total_reward)
            print("epoch done")
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
    FLAGS.use_convolution = True
    tf.app.run(main=main_fun, argv=[sys.argv[0]] + unparsed)
