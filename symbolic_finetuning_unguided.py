import pickle

import gym
import numpy as np
import retro

from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
from wrappers.circuscharlie import SimpleCircusCharlieWrapper  # noqa
from wrappers.pong import SimplePongWrapper  # noqa
from wrappers.seaquest import SimpleSeaquestWrapper


def to_discrete(raw_action, size=6):
    scaled_action = size / (1 + np.exp(raw_action))
    if scaled_action == size:
        return int(scaled_action) - 1
    else:
        return int(scaled_action)


def to_continuous(raw_action, low, high):
    scaled_action = (high - low) / (1 + np.exp(raw_action)) + low
    return np.float32(scaled_action)


def get_rewards(program, env_, render=False, assert_torch=False):
    done = False
    env = env_
    rewards = []
    for _ in range(4):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            raw_action = program.execute(np.expand_dims(obs, axis=0))
            if isinstance(env.action_space, gym.spaces.discrete.Discrete):
                action = to_discrete(raw_action, env.action_space.n)
            elif isinstance(env.action_space, gym.spaces.box.Box):
                action = to_continuous(raw_action, env.action_space.low, env.action_space.high)
            else:
                raise NotImplementedError("Only Discrete or Box action spaces are supported currently!")
            if assert_torch:
                raw_action_torch = program.execute_torch(np.expand_dims(obs, axis=0))
                action_torch = np.tanh(raw_action_torch.detach().numpy()).astype(np.float32)
                assert np.isclose(action_torch, action, atol=1e-4)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        rewards.append(episode_reward)
    # env.close()
    return np.mean(rewards)


rewards = make_fitness(get_rewards, greater_is_better=True, skip_checks=True)


def main():
    # env = gym.make("CartPole-v1")
    # env = gym.make("MountainCarContinuous-v0")
    # env = SimplePongWrapper(gym.make("PongNoFrameskip-v4"))
    # env = SimpleCircusCharlieWrapper(retro.make("CircusCharlie-Nes"))
    env = SimpleSeaquestWrapper(retro.make("Seaquest-Atari2600"))

    est_gp = SymbolicRegressor(
        population_size=16,
        generations=100,
        stopping_criteria=1000,
        metric=rewards,
        verbose=1,
        n_jobs=64,
        p_crossover=0.9,
        p_constants_sgd=0,
        # parsimony_coefficient=0.0001,
        init_depth=(2, 8),
    )

    try:
        est_gp.fit(env=env)
    except KeyboardInterrupt:
        pass
    # print(est_gp._program)
    # final_rewards = get_rewards(est_gp._program, env, render=False, assert_torch=False)
    # print(final_rewards)
    with open("circuscharlie.pkl", "wb") as f:
        pickle.dump(est_gp, f)


if __name__ == "__main__":
    main()
