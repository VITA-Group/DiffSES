import cv2
import gym
import numpy as np


class SimplePongWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SimplePongWrapper, self).__init__(env)
        print(f"{'=' * 10} Hacker Scripts have been Wrapped! {'=' * 10}")

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        self.pong_color = [236, 236, 236]
        self.player_color = [92, 186, 92]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        cropped = observation[34:-16, ...]  # Top cropping
        pong_attr = self.find_pong(cropped)
        player_attr = self.find_player(cropped)

        return np.array([*pong_attr, *player_attr]) / 160

    def step(self, action):
        # Super call
        observation, reward, done, info = self.env.step(action)
        cropped = observation[34:-16, ...]  # Top cropping
        pong_attr = self.find_pong(cropped)
        player_attr = self.find_player(cropped)
        info["obs"] = observation

        return np.array([*pong_attr, *player_attr]) / 160, reward, done, info

    def find_pong(self, observation):
        thresh = cv2.inRange(observation, np.array(self.pong_color), np.array(self.pong_color))
        contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        try:
            moments = cv2.moments(contour[0][0])
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        except (IndexError, ZeroDivisionError):
            # This color is not there in the image
            center_x = 0
            center_y = 0

        return center_x, center_y

    def find_player(self, observation):
        thresh = cv2.inRange(observation, np.array(self.player_color), np.array(self.player_color))
        contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        try:
            moments = cv2.moments(contour[0][0])
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        except (IndexError, ZeroDivisionError):
            # This color is not there in the image
            center_x = 0
            center_y = 0

        return center_x, center_y
