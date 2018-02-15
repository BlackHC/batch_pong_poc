import gym
import gym.spaces
import numpy as np
from src import vectorized_envs

class PongTiles:
    EMPTY = 0
    PADDLE = 1
    BALL = 2


class GameState:
    PLAYING = 0
    WIN_LEFT = 1
    WIN_RIGHT = 2


class GymAction:
    NOOP = 0
    UP = 1
    DOWN = 2


pong_state_dtype = np.dtype(
    [('paddle_centers', np.int8, 2), ('ball_position', np.float16, 2),
     ('ball_velocity', np.float16, 2), ('game_state', np.int8)])


class PongConfiguration(object):
    """
    Simple, batched pong environment written in Numpy.

    It doesn't exactly match the game mechanics from Atari Pong.
    """

    def __init__(self, width=40, height=30, paddle_half_height=2, paddle_step=2, ball_x_speed=1., ball_max_y_speed=3.):
        super().__init__()

        self.WIDTH = width
        self.HEIGHT = height
        # Actually PADDLE_HEIGHT = 2*PADDLE_HALF_HEIGHT + 1
        self.PADDLE_HALF_HEIGHT = paddle_half_height
        self.PADDLE_STEP = paddle_step
        self.BALL_X_SPEED = ball_x_speed
        self.BALL_MAX_Y_SPEED = ball_max_y_speed

        self.INITIAL_PONG_STATE = np.rec.array(
            [((self.HEIGHT / 2, self.HEIGHT / 2), (self.HEIGHT / 2, self.WIDTH / 2),
              (0, ball_x_speed), GameState.PLAYING)],
            dtype=pong_state_dtype)

    def create_batch_pong_state(self, batch_size):
        # We can support batches
        batch_pong_state = np.recarray(batch_size, dtype=pong_state_dtype)
        batch_pong_state[:] = self.INITIAL_PONG_STATE
        return batch_pong_state

    def handle_game_end(self, batch_pong_state):
        batch_pong_state[
            batch_pong_state.game_state != GameState.PLAYING] = self.INITIAL_PONG_STATE

    def update_game_state(self, batch_pong_state):
        batch_pong_state.game_state[
            batch_pong_state.ball_position[:, 1] < 0] = GameState.WIN_RIGHT
        batch_pong_state.game_state[
            batch_pong_state.ball_position[:, 1] >= self.WIDTH] = GameState.WIN_LEFT

    def reflect_on_borders(self, batch_pong_state):
        top_hit = batch_pong_state.ball_position[:, 0] < 0
        bottom_hit = batch_pong_state.ball_position[:, 0] >= self.HEIGHT

        batch_pong_state.ball_position[top_hit, 0] *= -1
        batch_pong_state.ball_position[bottom_hit, 0] = 2 * (
            self.HEIGHT - 1) - batch_pong_state[bottom_hit].ball_position[:, 0]
        batch_pong_state.ball_velocity[np.logical_or(top_hit, bottom_hit), 0] *= -1

    def reflect_on_paddle(self, batch_pong_state):
        # TODO(blackc): Reflection is not computed correctly.
        left_paddle_reflected_x = 2 * 1 - batch_pong_state.ball_position[:, 1]
        right_paddle_reflected_x = 2 * (
            self.WIDTH - 2) - batch_pong_state.ball_position[:, 1]

        paddle_height_distance = (
            batch_pong_state.ball_position[:, 0][:, np.newaxis] -
            batch_pong_state.paddle_centers)

        left_paddle_hit = np.logical_and(
            left_paddle_reflected_x >= 1,
            np.fabs(paddle_height_distance[:, 0]) <= self.PADDLE_HALF_HEIGHT)

        right_paddle_hit = np.logical_and(
            right_paddle_reflected_x < self.WIDTH - 2,
            np.fabs(paddle_height_distance[:, 1]) < self.PADDLE_HALF_HEIGHT + 1)

        batch_pong_state.ball_position[
            left_paddle_hit, 1] = left_paddle_reflected_x[left_paddle_hit]
        batch_pong_state.ball_position[
            right_paddle_hit, 1] = right_paddle_reflected_x[right_paddle_hit]

        paddle_hit = np.logical_or(left_paddle_hit, right_paddle_hit)
        batch_pong_state.ball_velocity[
            paddle_hit, 1] = -batch_pong_state.ball_velocity[paddle_hit, 1]

        batch_pong_state.ball_velocity[left_paddle_hit, 0] = np.clip(
            batch_pong_state.ball_velocity[left_paddle_hit, 0] +
            paddle_height_distance[left_paddle_hit, 0] / self.PADDLE_HALF_HEIGHT,
            -self.BALL_MAX_Y_SPEED, self.BALL_MAX_Y_SPEED)

        batch_pong_state.ball_velocity[right_paddle_hit, 0] = np.clip(
            batch_pong_state.ball_velocity[right_paddle_hit, 0] +
            paddle_height_distance[right_paddle_hit, 1] / self.PADDLE_HALF_HEIGHT,
            -self.BALL_MAX_Y_SPEED, self.BALL_MAX_Y_SPEED)

    def move_ball(self, batch_pong_state):
        batch_pong_state.ball_position += batch_pong_state.ball_velocity

    def render_pong_states(self, batch_pong_state):
        batch_bitmap = np.zeros(
            (batch_pong_state.shape[0], self.HEIGHT, self.WIDTH), dtype=np.uint8)
        batch_ball_indices = np.floor(
            batch_pong_state.ball_position).astype(dtype=np.int)
        batch_ball_indices[:, 0] = np.clip(batch_ball_indices[:, 0], 0, self.HEIGHT - 1)
        batch_ball_indices[:, 1] = np.clip(batch_ball_indices[:, 1], 0, self.WIDTH - 1)

        batch_bitmap[
            list(range(batch_ball_indices.shape[0])),
            batch_ball_indices[:, 0],
            batch_ball_indices[:, 1]] = PongTiles.BALL

        paddle_range = np.arange(
            -self.PADDLE_HALF_HEIGHT,
            self.PADDLE_HALF_HEIGHT + 1)[np.newaxis, np.newaxis, :]

        batch_paddle_indices = np.floor(batch_pong_state.paddle_centers).astype(
            np.int)
        batch_expanded_paddles = np.clip(
            batch_paddle_indices[..., np.newaxis] + paddle_range,
            0,
            self.HEIGHT - 1)

        batch_bitmap[
            np.array(list(range(batch_ball_indices.shape[0])))[:, np.newaxis, np.newaxis],
            batch_expanded_paddles,
            np.array([0, self.WIDTH - 1]).reshape((1, 2, 1))] = PongTiles.PADDLE

        return batch_bitmap

    def move_paddles(self, batch_pong_state, paddle_step):
        np.clip(
            batch_pong_state.paddle_centers + paddle_step,
            -self.PADDLE_HALF_HEIGHT * 2,
            self.PADDLE_HALF_HEIGHT * 2 + self.HEIGHT,
            batch_pong_state.paddle_centers)

    def pong_step(self, batch_pong_state, batch_paddle_direction):
        assert (np.all(np.abs(batch_paddle_direction) <= 1))

        self.move_paddles(batch_pong_state, batch_paddle_direction * self.PADDLE_STEP)
        self.move_ball(batch_pong_state)
        self.reflect_on_borders(batch_pong_state)
        self.reflect_on_paddle(batch_pong_state)

        self.update_game_state(batch_pong_state)


# Supports OpenAI Universe's vectorized interface
class BothPlayerPongGymEnv(gym.Env):
    metadata = {'render.modes': ['text_block', 'rgb_array', 'human']}

    def __init__(self, pong_config: PongConfiguration = PongConfiguration(), n=1):
        super().__init__()

        self.n = n
        self.pong_config = pong_config

        self.action_space = gym.spaces.MultiDiscrete([(0, 3), (0, 3)])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(pong_config.HEIGHT, pong_config.HEIGHT))
        # We can keep the default reward_range.

        # API requires a call to reset() before step() can be called.
        self.pong_state = None
        self.outputs = None

    def _step(self, actions):
        infos = ((),) * self.n
        # A bit awkward but I don't have an AI player, so I'll just make the agents
        # maximize the #steps in the game.
        # And play both agents.
        paddle_direction = -1 * (actions == GymAction.UP) + 1 * (actions == GymAction.DOWN)
        self.pong_config.pong_step(self.pong_state, paddle_direction)

        rewards = 1 * (self.pong_state.game_state == GameState.PLAYING)
        dones = self.pong_state.game_state != GameState.PLAYING

        self.pong_config.handle_game_end(self.pong_state)
        self.outputs = self.pong_config.render_pong_states(self.pong_state)

        return self.outputs, rewards, dones, infos

    def _reset(self):
        self.pong_state = self.pong_config.create_batch_pong_state(self.n)
        self.outputs = self.pong_config.render_pong_states(self.pong_state)

        return self.pong_config.render_pong_states(self.pong_state)

    def _render(self, mode='rgb_array', close=False):
        if close:
            return

        if mode == 'text_block':
            return render_array_as_text(('.', 'P', 'B'), self.outputs[0])
        elif mode == 'rgb_array':
            return render_array_as_bitmap(((0, 0, 0), (0, 0, 255), (0, 0, 255)), self.outputs[0])
        else:
            raise ValueError('Unknown mode \'%s\'' % mode)

    def _seed(self, seed=None):
        # TODO(blackhc): Add random direction to ball in create_batch_pong_state.
        pass

    def get_keys_to_action(self):
        return {
            (): GymAction.NOOP,
            (ord('w'),): GymAction.UP,
            (ord('s'),): GymAction.DOWN,
        }


def render_array_as_text(chars, bm):
    chars = np.array(list(chars))
    return ''.join([''.join(row) + '\n' for row in chars[bm]])


def render_array_as_bitmap(values, bm):
    values = np.array(list(values))
    return values[bm, :]


if __name__ == '__main__':
    import gym.utils.play as play

    env = vectorized_envs.UnvectorizeGymEnv(BothPlayerPongGymEnv())
    play.play(env, zoom=20, fps=10)
