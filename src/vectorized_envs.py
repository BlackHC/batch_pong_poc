# batch_pong_poc
# Copyright (C) 2018  Andreas blackhc@ Kirsch
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import gym


class UnvectorizeGymEnv(gym.Wrapper):
    def _step(self, actions):
        obss, rewards, dones, infos = self.env.step(actions)
        return obss[0], rewards[0], dones[0], infos[0]

    def _reset(self):
        return self.env.reset()[0]

    def _render(self, mode='rgb_array', close=False):
        if close:
            return self.env.render(mode, close)

        return self.env.render(mode, close)[0]


class SequentiallyVectorizedEnv(gym.Env):
    def __init__(self, envs):
        self.envs = envs
        self.n = len(envs)

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def reward_range(self):
        return self.envs[0].reward_range

    @property
    def metadata(self):
        return self.envs[0].metadata

    def _step(self, actions):
        results = map(lambda env, action: env.step(action), self.envs, actions)
        obss, rewards, dones, infos = zip(*results)
        obss = [obs if not done else env.reset() for obs, env, done in zip(obss, self.envs, dones)]
        return obss, rewards, dones, infos

    def _reset(self):
        return [env.reset() for env in self.envs]

    def _render(self, mode='rgb_array', close=False):
        if close:
            for env in self.envs:
                env.render(mode, close)
            return

        return [env.render(mode, close) for env in self.envs]


def make_gym_sequential_envs(env_id, n=1):
    return SequentiallyVectorizedEnv([gym.make(env_id) for _ in range(n)])
