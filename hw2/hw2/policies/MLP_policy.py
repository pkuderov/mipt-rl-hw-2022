import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cds_rl.infrastructure import pytorch_util as ptu
from cds_rl.policies.base_policy import BasePolicy
from hw2.infrastructure.utils import normalize, unnormalize


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        action = self(torch.Tensor(observation).to(ptu.device))
        return action.cpu().detach().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            return self.logits_na(observation)
        return self.mean_net(observation)


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss

        self.optimizer.zero_grad(set_to_none=True)
        output = self(torch.Tensor(observations).to(ptu.device))

        loss = self.loss(output, torch.Tensor(actions).to(ptu.device))

        loss.backward()
        self.optimizer.step()
        
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

    class MLPPolicyPG(MLPPolicy):
        def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
            super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
            self.baseline_loss = nn.MSELoss()

        def update(self, observations, actions, advantages, q_values=None):
            observations = ptu.from_numpy(observations)
            actions = ptu.from_numpy(actions)
            advantages = ptu.from_numpy(advantages)

            # TODO: update the policy using policy gradient
            # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
            # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
            # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
            # HINT4: use self.optimizer to optimize the loss. Remember to
            # 'zero_grad' first

            self.optimizer.zero_grad()
            forward_return = self.forward(observations)
            log_prob = forward_return.log_prob(actions)
            loss = torch.sum(-log_prob * advantages)
            loss.backward()

            if self.nn_baseline:
                ## TODO: update the neural network baseline using the q_values as
                ## targets. The q_values should first be normalized to have a mean
                ## of zero and a standard deviation of one.

                ## HINT1: use self.baseline_optimizer to optimize the loss used for
                ## updating the baseline. Remember to 'zero_grad' first
                ## HINT2: You will need to convert the targets into a tensor using
                ## ptu.from_numpy before using it in the loss

                self.baseline_optimizer.zero_grad()
                baseline_pred = self.baseline(observations).squeeze()
                baseline_target = ptu.from_numpy(normalize(q_values, np.mean(q_values), np.mean(q_values)))
                baseline_loss = self.baseline_loss(baseline_pred, baseline_target)
                baseline_loss.backward()
                self.baseline_optimizer.step()

            self.optimizer.step()

            train_log = {
                'Training Loss': ptu.to_numpy(loss),
            }
            return train_log

        def run_baseline_prediction(self, observations):
            """
                Helper function that converts `observations` to a tensor,
                calls the forward method of the baseline MLP,
                and returns a np array
                Input: `observations`: np.ndarray of size [N, 1]
                Output: np.ndarray of size [N]
            """
            observations = ptu.from_numpy(observations)
            pred = self.baseline(observations)
            return ptu.to_numpy(pred.squeeze())
