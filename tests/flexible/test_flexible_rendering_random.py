import unittest

import gym
import imageio
import numpy as np


class TestRendering(unittest.TestCase):

    def test_random_mt10c1_rendering(self):
        env = gym.make('JSSEnv:flexible-jss-v1', env_config={'instance_path': '../../JSSEnv/envs/instances/flexible/mt10c1.fjs'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        done = False
        step_nb = 0
        images = []
        while not done:
            legal_actions = env.get_legal_actions()
            actions = np.random.choice(len(legal_actions), 1, p=(legal_actions / legal_actions.sum()))[0]
            state, reward, done, _ = env.step(actions)
            temp_image = env.render().to_image()
            images.append(imageio.imread(temp_image))
        imageio.mimsave("mt10c1.gif", images)
        env.reset()
        self.assertEqual(env.current_time_step, 0)