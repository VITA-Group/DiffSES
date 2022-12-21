import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack

from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
from wrappers.sb3_patches import make_atari_env

ACTION_INDEX = 0
N_ENVS = 4
PPO_MODEL = PPO.load("logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip")


def to_discrete(raw_action, obs, size):
    ppo_model_action, _ = PPO_MODEL.predict(obs)
    action_logits = np.zeros(size, dtype=np.float32)
    action_logits[int(ppo_model_action)] = 1
    action_logits[ACTION_INDEX] = raw_action
    action_distribution = np.exp(action_logits) / np.sum(np.exp(action_logits))
    action = np.argmax(action_distribution)
    return action


def get_rewards(program, env_, render=False):
    done = False
    env = env_
    rewards = []
    for _ in range(1):
        obs = env.reset()
        done = np.array([False] * N_ENVS)
        episode_reward = np.zeros(N_ENVS)
        info = None
        while not done.all():
            if info is None:
                action = env.action_space.sample()
            else:
                # od_result = od(info)
                # raw_action = program.execute(np.expand_dims(od_result, axis=0))
                raw_action = 0.3
                if isinstance(env.action_space, gym.spaces.discrete.Discrete):
                    action = to_discrete(raw_action, obs, env.action_space.n)
                elif isinstance(env.action_space, gym.spaces.box.Box):
                    raise NotImplementedError("Box action space not yet supported!")
                else:
                    raise NotImplementedError("Only Discrete or Box action spaces are supported currently!")
            obs, reward, done, info = env.step([action] * N_ENVS)
            episode_reward += reward
            if render:
                env.render()
        rewards.append(episode_reward)
    env.close()
    return np.mean(rewards)


rewards = make_fitness(get_rewards, greater_is_better=True, skip_checks=True)


def main():
    env = make_atari_env("PongNoFrameskip-v4", n_envs=N_ENVS)
    env = VecFrameStack(env, n_stack=4)
    # env = gym.make("PongNoFrameskip-v4")
    # env = ObservationToInfo(env)
    # env = AtariWrapper(env)
    # env = SqueezeObservation(env)
    # env = FrameStack(env, num_stack=4)

    est_gp = SymbolicRegressor(
        population_size=200,
        generations=100,
        stopping_criteria=21,
        metric=rewards,
        verbose=1,
        n_jobs=24,
        p_crossover=0.9,
        p_constants_sgd=0,
    )

    try:
        est_gp.fit(env=env)
    except KeyboardInterrupt:
        pass
    print(est_gp._program)
    final_rewards = get_rewards(est_gp._program, env, render=False)
    print(final_rewards)


if __name__ == "__main__":
    main()
