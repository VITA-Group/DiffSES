# /home/user/miniconda3/envs/symrel-v2/lib/python3.7/site-packages/stable_baselines3/common/policies_new.py
# needs to be changed
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class CustomNetwork(nn.Module):
    def __init__(self, program):
        super(CustomNetwork, self).__init__()

        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

        self.policy_net = program
        self.value_net = program

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


def main():
    model = PPO(CustomActorCriticPolicy, "CartPole-v0", verbose=1, policy_kwargs={"program": "amogus"})
    print(model.policy.mlp_extractor.policy_net)


class CustomActorCriticPolicy(ActorCriticPolicy, nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.Tanh,
        *args,
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # self.program = program

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.program)


if __name__ == "__main__":
    main()
