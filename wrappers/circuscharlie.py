import cv2
import gym
import numpy as np


class SimpleCircusCharlieWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SimpleCircusCharlieWrapper, self).__init__(env)
        # print(f"{'=' * 10} Hacker Scripts have been Wrapped! {'=' * 10}")

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,))
        self.action_space = gym.spaces.Discrete(9)
        # self.ring_color = [[248, 152, 56]]
        self.ring_color = [134]
        # self.charlie_color = [248, 188, 176]
        self.charlie_color = [191]
        # self.pot_color = [248, 252, 248]
        self.pot_color = [250]

    def step(self, action):
        # Super call
        action = np.identity(9)[action]
        observation, reward, done, info = self.env.step(action)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        charlie_attr = self.find_charlie(observation)
        ring_attr = self.find_ring(observation, charlie_attr)
        pot_attr = self.find_pot(observation)

        # cv2.circle(observation, charlie_attr, 10, [0], -1)
        # cv2.circle(observation, pot_attr, 10, [0], -1)
        # for x, y in zip(ring_attr[0::2], ring_attr[1::2]):
        #     cv2.circle(observation, [x, y], 6, [0], -1)
        info["obs"] = observation

        return np.array([*charlie_attr, *ring_attr, *pot_attr]) / 240, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        charlie_attr = self.find_charlie(observation)
        ring_attr = self.find_ring(observation, charlie_attr)
        pot_attr = self.find_pot(observation)

        return np.array([*charlie_attr, *ring_attr, *pot_attr]) / 240

    def find_charlie(self, observation):
        thresh = cv2.inRange(observation, np.array(self.charlie_color), np.array(self.charlie_color))
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

    def find_ring(self, observation, charlie_attr):
        thresh = np.zeros((224, 240), np.uint8)
        for each_type in self.ring_color:
            thresh += cv2.inRange(observation, np.array(each_type), np.array(each_type))
        thresh = cv2.dilate(thresh, np.ones((13, 13)))
        contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        all_rings = []
        for each_ring in contour[0]:
            if cv2.contourArea(each_ring) == 0:
                continue
            try:
                moments = cv2.moments(each_ring)
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            except (IndexError, ZeroDivisionError, ValueError):
                center_x = 0
                center_y = 0
            all_rings.append([center_x, center_y])

        if len(all_rings):
            closest_ring_idx = np.argsort(np.abs(np.array(all_rings) - np.array(charlie_attr))[:, 0])[:2]
            all_rings = np.array(all_rings)[closest_ring_idx]
            if len(all_rings) > 2:
                all_rings = all_rings[:2]

        return self.flatten_and_pad_to_length(all_rings, 4).astype(np.int)

    def find_pot(self, observation):
        mask = np.zeros((224, 240))
        mask[170:205, ...] = 1
        thresh = cv2.inRange(mask * observation, np.array(self.pot_color), np.array(self.pot_color))
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

    @staticmethod
    def flatten_and_pad_to_length(array: list, length: int = 12):
        flattened = np.array(array).flatten()
        npad = length - len(flattened)
        padded = np.pad(flattened, pad_width=npad, mode="constant", constant_values=0)[npad:]

        return padded
