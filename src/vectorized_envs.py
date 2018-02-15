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
