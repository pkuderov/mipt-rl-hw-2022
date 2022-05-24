import numpy as np
import torch

from hw3.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw3.policies.argmax_policy import ArgMaxPolicy
from hw3.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        perform_random_action = (self.t < self.learning_starts) or (np.random.random() < eps)

        if perform_random_action:
            action = np.random.randint(self.num_actions)
        else:
            enc_last_obs = self.replay_buffer.encode_recent_observation()
            enc_last_obs = torch.tensor(enc_last_obs[None, :]).to(self.device)

            # TODO query the policy with enc_last_obs to select action
            action = self.actor.get_action(enc_last_obs)

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see replay buffer's store_effect function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = obs

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        """
            Here, you should train the DQN agent.
            This consists of training the critic, as well as periodically updating the target network.
        """
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

            # TODO: load newest parameters into the target network
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.target_Q_func.load_state_dict(self.critic.Q_func.state_dict())

            self.num_param_updates += 1

        self.t += 1

        return log
