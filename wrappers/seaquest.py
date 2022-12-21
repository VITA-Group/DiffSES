import cv2
import gym
import numpy as np


class SimpleSeaquestWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SimpleSeaquestWrapper, self).__init__(env)
        # print(f"{'=' * 10} Hacker Scripts have been Wrapped! {'=' * 10}")

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.action_space = gym.spaces.Discrete(8)
        self.protagonist_color = [187, 187, 53]
        self.fish_color = [[92, 186, 92], [160, 171, 79], [170, 170, 170]]  # Last one is submarine

    def step(self, action):
        action = np.identity(8)[action]
        # Super call
        observation, reward, done, info = self.env.step(action)
        # observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        protagonist_attr = self.find_protagonist(observation)
        fish_attr = self.find_fish(observation, protagonist_attr)  # Gets closest 5 fish

        # cv2.circle(observation, protagonist_attr, 10, [255, 0, 0])
        # for x, y in zip(fish_attr[0::2], fish_attr[1::2]):
        #     cv2.circle(observation, [x, y], 6, [0, 255, 0])
        info["obs"] = observation

        # return np.array([*protagonist_attr, *fish_attr]) / 160, reward, done, info
        return np.array([*protagonist_attr, *fish_attr]) / 160, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        protagonist_attr = self.find_protagonist(observation)
        fish_attr = self.find_fish(observation, protagonist_attr)  # Gets closest 5 fish

        return np.array([*protagonist_attr, *fish_attr]) / 160

    def find_protagonist(self, observation):
        thresh = cv2.inRange(observation, np.array(self.protagonist_color) - 2, np.array(self.protagonist_color) + 2)
        contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        try:
            moments = cv2.moments(max(contour[0], key=cv2.contourArea))
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        except (IndexError, ZeroDivisionError, ValueError):
            # This color is not there in the image
            center_x = 0
            center_y = 0

        return center_x, center_y

    def find_fish(self, observation, protagonist_attr):
        thresh = np.zeros((210, 160), np.uint8)
        for each_type in self.fish_color:
            thresh += cv2.inRange(observation, np.array(each_type), np.array(each_type))
        contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        all_fish = []
        for each_fish in contour[0]:
            if cv2.contourArea(each_fish) == 0:
                continue
            try:
                moments = cv2.moments(each_fish)
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            except (IndexError, ZeroDivisionError, ValueError):
                center_x = 0
                center_y = 0
            all_fish.append([center_x, center_y])

        if len(all_fish):
            closest_fish_idx = np.argsort(np.abs(np.array(all_fish) - np.array(protagonist_attr))[:, 0])[:4]
            all_fish = np.array(all_fish)[closest_fish_idx]
            if len(all_fish) > 4:
                all_fish = all_fish[:4]

        return self.flatten_and_pad_to_length(all_fish, 8).astype(np.int)

    @staticmethod
    def flatten_and_pad_to_length(array: list, length: int = 12):
        flattened = np.array(array).flatten()
        npad = length - len(flattened)
        padded = np.pad(flattened, pad_width=npad, mode="constant", constant_values=0)[npad:]

        return padded
