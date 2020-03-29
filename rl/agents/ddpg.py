from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001,
                 tb_log_dir=None, tb_full_log=False, log_freq=10, **kwargs):
        if hasattr(actor.outputs, '__len__') and len(actor.outputs) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.outputs, '__len__') and len(critic.outputs) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory = memory

        self.actor_updates = []
        self.critic_updates = []

        # State.
        self.compiled = False
        self.reset_states()

        # tensorflow log
        self.tb_log_dir = tb_log_dir
        self.tb_full_log = tb_full_log
        self.log_freq = log_freq

    @property
    def tb_log_dir(self):
        return self._tb_log_dir

    @tb_log_dir.setter
    def tb_log_dir(self, value):
        if value is not None and isinstance(value, str):
            self._tb_log_dir = value
            train_log_dir = os.path.abspath(os.path.join(self.tb_log_dir, 'train'))
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            self._tb_log_dir = None
            self.train_summary_writer = None

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]
        print(optimizer._name)

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using tensorflow.keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            self.critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            #critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        if self.target_model_update < 1.:
            # Include soft target model updates.
            actor_updates = get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)

        self.actor_updates = actor_updates
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.compiled = True

    @tf.function
    def critic_train_fn(self, actions, state_inputs, true_reward):

        def clipped_error(y_true, y_pred):
            # Todo: mean over all or axis=-1?
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip))#, axis=-1)

        with tf.GradientTape() as tape:
            reward = self.critic([actions, state_inputs])
            loss = clipped_error(true_reward, reward)

        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        return reward, loss

    @tf.function
    def actor_train_fn(self, state_inputs):

        with tf.GradientTape() as tape:
            actions = self.actor(state_inputs)
            reward = self.target_critic([actions, state_inputs])
            loss = - tf.reduce_mean(reward)

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return actions, loss

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self._update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    @tf.function
    def _update_target_models(self):
        # apply all updates to the actor target
        for target, value_op in self.critic_updates:
            K.update(target, value_op)

        # apply all updates to the actor target
        for target, value_op in self.actor_updates:
            K.update(target, value_op)

    def _update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    #@tf.function
    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = tf.reshape(self.actor(batch), [self.nb_actions, ])
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action.numpy()

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        # check if actor and critic exists before returnin gtheir layers
        if hasattr(self, 'actor') and hasattr(self, 'critic'):
            return self.actor.layers[:] + self.critic.layers[:]
        else:
            return []

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # check if warm time is over
            if self.step > self.nb_steps_warmup_critic:
                # Update critic
                # -----------------------
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = tf.reshape(self.target_critic.predict_on_batch(state1_batch_with_action),
                                             [self.batch_size,])
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = tf.reshape((reward_batch + discounted_reward_batch), (self.batch_size, 1))

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)

                #metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                predicted_reward, critic_loss = self.critic_train_fn(*state0_batch_with_action, targets)

                # Todo: see what to do with these dummy metrics
                metrics = [np.nan for _ in self.metrics_names]
                if self.processor is not None:
                    metrics += self.processor.metrics

                # Update actor
                # ---------------------------------------
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                # Todo: find out what this did!
                #if self.uses_learning_phase:
                #    inputs += [self.training]
                action_values, action_loss = self.actor_train_fn(inputs)
                assert action_values.shape == (self.batch_size, self.nb_actions)

                # update target networks
                self._update_target_models()

                # log to tensorboard
                if self.tb_log_dir and self.step % self.log_freq == 0:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('reward', reward, step=self.step)
                        tf.summary.scalar('actor_loss', action_loss, step=self.step)
                        tf.summary.scalar('critic_loss', critic_loss, step=self.step)

                        if self.tb_full_log:
                            pass

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self._update_target_models_hard()

        return metrics
