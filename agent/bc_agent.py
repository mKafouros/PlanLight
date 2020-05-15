from . import BaseAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
import tensorflow as tf

class Policy_net:
    def __init__(self, name: str, ob_space, act_space):
        """
        :param name: string
        :param ob_space:
        :param act_space:
        """

        # ob_space = env.observation_space
        # act_space = env.action_space

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space), name='obs')

            with tf.variable_scope('policy_net', reuse=tf.AUTO_REUSE):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net', reuse=tf.AUTO_REUSE):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)



class BehavioralCloning:
    def __init__(self, Policy):

        self.Policy = Policy

        self.actions_expert = tf.placeholder(tf.int32, shape=[None], name='actions_expert')

        actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)

        loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        loss = - tf.reduce_mean(loss)
        tf.summary.scalar('loss/cross_entropy', loss)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)

        self.merged = tf.summary.merge_all()
        self.loss = loss

    def train(self, obs, actions):
        return tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                                      self.actions_expert: actions})

    def get_summary(self, obs, actions):
        return tf.get_default_session().run(self.loss, feed_dict={self.Policy.obs: obs,
                                                                    self.actions_expert: actions})


class BehaviorCloningAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, iid, args, name="policy"):
        super().__init__(action_space)

        self.iid = iid

        self.args = args

        self.ob_space = ob_generator.ob_space()
        self.ob_generator = ob_generator
        self.action_space = action_space
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.Policy = Policy_net('{}_{}'.format(name, iid), self.ob_space, self.action_space)
        self.bc = BehavioralCloning(self.Policy)
        self.writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=None)


    def get_reward(self):
        return 1.


    def update_policy(self, obs, actions):
        with self.sess.as_default():
            obs = np.reshape(obs, newshape=[-1] + list(self.ob_space))
            self.bc.train(obs, actions)
            loss = self.bc.get_summary(obs, actions)
        return loss

    def get_action(self, ob, stochastic=False):
        with self.sess.as_default():
            obs = np.stack([ob]).astype(dtype=np.float32)
            act, _ = self.Policy.act(obs=obs, stochastic=stochastic)

        return act[0]

    def get_actions(self, obs, stochastic=False):
        with self.sess.as_default():
            obs = np.asarray(obs).reshape(-1, self.ob_space[0])
            act, _ = self.Policy.act(obs=obs, stochastic=stochastic)

        return act

    def sample(self):
        return self.action_space.sample()

    def get_ob(self):
        return self.ob_generator.generate()


    def load_model(self, dir="model/bc", model_id=None):
        if model_id is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(dir))
        self.saver.restore(self.sess, dir+"-"+str(model_id))

    def save_model(self, step, dir="model/bc"):
        self.saver.save(self.sess, dir, global_step=step)


    # def __del__(self):
    #     self.sess.close()