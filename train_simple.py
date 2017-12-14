import argparse
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_paneldepon.env import register
from gym_paneldepon.util import print_up

from util import bias_variable, summarize_scalar, variable_summaries, vh_log, weight_variable  # noqa: I001

register()


FLAGS = None


class Agent(object):
    BATCH_SIZE = 1
    HIDDEN_SIZE = 200

    def __init__(self):
        self.env = gym.make('PdPEndless4-v0')
        self.make_graph()
        self.make_summaries()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(tf.get_default_graph())

    def make_graph(self):
        self.make_input_graph()
        self.make_hidden_graph()
        self.make_output_graph()
        self.make_loss_graph()
        self.make_train_graph()

    def make_summaries(self):
        variable_summaries(self.W_hidden, "W_hidden")
        variable_summaries(self.b_hidden, "b_hidden")
        variable_summaries(self.W_output, "W_output")
        variable_summaries(self.b_output, "b_output")

        tf.summary.histogram('Q', self.output)

    def make_input_graph(self):
        chain_space, box_space = self.env.observation_space.spaces
        with tf.name_scope("input"):
            self.chain_input = tf.placeholder(tf.float32, [self.BATCH_SIZE, 1], name="chain")
            self.box_input = tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(box_space.shape), name="box")
        self.n_inputs = np.prod(box_space.shape) + 1

    def make_hidden_graph(self):
        with tf.name_scope("flatten"):
            flat_input = tf.reshape(self.box_input, [-1, self.n_inputs - 1])
            flat_input = tf.concat([flat_input, self.chain_input], 1)

        with tf.name_scope("hidden"):
            self.W_hidden = weight_variable([self.n_inputs, self.HIDDEN_SIZE], name="W")
            self.b_hidden = bias_variable([self.HIDDEN_SIZE], name="b")
            z = tf.matmul(flat_input, self.W_hidden) + self.b_hidden
            self.hidden_activation = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_outputs = self.env.action_space.n
        with tf.name_scope("output"):
            self.W_output = weight_variable([self.HIDDEN_SIZE, self.n_outputs], name="W")
            self.b_output = bias_variable([self.n_outputs], name="b")
            z = tf.matmul(self.hidden_activation, self.W_output) + self.b_output
            self.output = z
            self.action = tf.argmax(self.output, 1)

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_outputs], name="target")
            with tf.name_scope("error"):
                self.error = tf.reduce_sum(tf.square(self.output - self.target))
            with tf.name_scope("L2-norm"):
                self.L2_norm = tf.reduce_sum(tf.square(self.b_hidden)) + tf.reduce_sum(tf.square(self.b_output))
                self.L2_norm += tf.reduce_sum(tf.square(self.W_hidden)) + tf.reduce_sum(tf.square(self.W_output))
            self.loss = self.error + self.L2_norm * 1e-6

    def make_train_graph(self):
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, state):
        chain, box = state
        return {self.chain_input: [[chain]], self.box_input: [box]}

    def render_in_place(self):
        self.env.render()
        print_up(5)

    def render_ansi(self):
        sio = self.env.render("ansi")
        print_up(5, outfile=sio)
        return sio

    def dump(self, session):
        outputs_dir = os.getenv('VH_OUTPUTS_DIR', '/tmp/tensorflow/gym_paneldepon/outputs')
        arrays = session.run([self.W_hidden, self.b_hidden, self.W_output, self.b_output])
        for arr, name in zip(arrays, ["W_hidden", "b_hidden", "W_output", "b_output"]):
            filename = os.path.join(outputs_dir, "{}.csv".format(name))
            np.savetxt(filename, arr, delimiter=",")


def main(*args, **kwargs):
    gamma = 0.99
    agent = Agent()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_episodes):
            exploration = 1 / (0.1 * i + 2)
            vh_log({"exploration": exploration}, i)
            total_reward = 0
            state = agent.env.reset()
            # Fully raise the panel stack
            for k in range(4):
                state, _, _, _ = agent.env.step(1)
            frames = []
            for j in range(FLAGS.num_steps):
                action, Q_base = sess.run([agent.action, agent.output], feed_dict=agent.get_feed_dict(state))
                if np.random.rand(1) < exploration:
                    action[0] = agent.env.action_space.sample()
                new_state, reward, _, _ = agent.env.step(action[0])
                Q = sess.run(agent.output, feed_dict=agent.get_feed_dict(new_state))  # noqa: N806
                Q_target = Q_base  # noqa: N806
                Q_target[0, action[0]] = reward + gamma * np.max(Q)
                if FLAGS.do_render:
                    # We only render 10% from the start of the episode so as not to clog the console.
                    frames.append(agent.render_ansi().getvalue())
                    if j % 10 == 0:
                        print(frames.pop(0), end="")
                feed_dict = agent.get_feed_dict(state)
                feed_dict[agent.target] = Q_target
                sess.run(agent.train_step, feed_dict=feed_dict)

                total_reward += reward
                state = new_state
            if FLAGS.do_render:
                for _ in range(5):
                    print()
            vh_log({"reward": total_reward}, i)
            summarize_scalar(agent.writer, "Reward", total_reward, i)
            summary = sess.run(merged, feed_dict=feed_dict)
            agent.writer.add_summary(summary, i)
        agent.dump(sess)
    agent.writer.close()


def main_with_render(*args, **kwargs):
    try:
        print("\033[?25l")
        main(*args, **kwargs)
    finally:
        print("\033[?25h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=300,
                        help='Number of episodes to run the trainer')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of steps per episode')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/gym_paneldepon/logs/rl_with_summaries',
                        help='Summaries log directory')
    parser.add_argument('--no_render', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.do_render = not FLAGS.no_render
    main_fun = main_with_render if FLAGS.do_render else main
    tf.app.run(main=main_fun, argv=[sys.argv[0]] + unparsed)
