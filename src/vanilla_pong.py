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


class PongGame(object):
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

        self.paddle_centers = None
        self.ball_position = None
        self.ball_velocity = None
        self.game_state = None

    def reset_game(self):
        self.paddle_centers = np.array([self.HEIGHT / 2, self.HEIGHT / 2])
        self.ball_position = np.array([self.HEIGHT / 2, self.WIDTH / 2])
        self.ball_velocity = np.array([0, self.BALL_X_SPEED])
        self.game_state = GameState.PLAYING

    def handle_game_end(self):
        if self.game_state != GameState.PLAYING:
            self.reset_game()

    def update_game_state(self):
        if self.ball_position[1] < 0:
            self.game_state = GameState.WIN_RIGHT
        if self.ball_position[1] >= self.WIDTH:
            self.game_state = GameState.WIN_LEFT

    def reflect_on_borders(self):
        if self.ball_position[0] < 0:
            self.ball_position[0] *= -1
            self.ball_velocity[0] *= -1

        if self.ball_position[0] >= self.HEIGHT:
            self.ball_position[0] = 2 * (
                    self.HEIGHT - 1) - self.ball_position[0]
            self.ball_velocity[0] *= -1

    def reflect_on_paddle(self):
        # TODO(blackc): Reflection is not computed correctly.
        left_paddle_reflected_x = 2 * 1 - self.ball_position[1]
        right_paddle_reflected_x = 2 * (
                self.WIDTH - 2) - self.ball_position[1]

        paddle_height_distance = (
                self.ball_position[np.newaxis, 0] -
                self.paddle_centers)

        left_paddle_hit = left_paddle_reflected_x >= 1 and np.fabs(paddle_height_distance[0]) <= self.PADDLE_HALF_HEIGHT
        right_paddle_hit = right_paddle_reflected_x < self.WIDTH - 2 and np.fabs(paddle_height_distance[1]) < self.PADDLE_HALF_HEIGHT + 1

        if left_paddle_hit:
            self.ball_position[1] = left_paddle_reflected_x

            self.ball_velocity[1] *= -1
            self.ball_velocity[0] = np.clip(
                self.ball_velocity[0] + paddle_height_distance[0] / self.PADDLE_HALF_HEIGHT,
                -self.BALL_MAX_Y_SPEED, self.BALL_MAX_Y_SPEED)

        if right_paddle_hit:
            self.ball_position[1] = right_paddle_reflected_x

            self.ball_velocity[1] *= -1
            self.ball_velocity[0] = np.clip(
                self.ball_velocity[0] + paddle_height_distance[1] / self.PADDLE_HALF_HEIGHT,
                -self.BALL_MAX_Y_SPEED, self.BALL_MAX_Y_SPEED)

    def move_ball(self):
        self.ball_position += self.ball_velocity

    def render_pong_states(self):
        bitmap = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        ball_index = np.floor(self.ball_position).astype(dtype=np.int)
        ball_index[0] = np.clip(ball_index[0], 0, self.HEIGHT - 1)
        ball_index[1] = np.clip(ball_index[1], 0, self.WIDTH - 1)

        bitmap[ball_index[0], ball_index[1]] = PongTiles.BALL

        paddle_range = np.arange(
            -self.PADDLE_HALF_HEIGHT,
            self.PADDLE_HALF_HEIGHT + 1)[np.newaxis, :]

        paddle_indices = np.floor(self.paddle_centers).astype(
            np.int)
        expanded_paddles = np.clip(
            paddle_indices[:, np.newaxis] + paddle_range,
            0,
            self.HEIGHT - 1)

        bitmap[
            expanded_paddles,
            np.array([0, self.WIDTH - 1]).reshape((2, 1))] = PongTiles.PADDLE

        return bitmap.reshape((1, self.HEIGHT, self.WIDTH))

    def move_paddles(self, paddle_step):
        np.clip(
            self.paddle_centers + paddle_step,
            -self.PADDLE_HALF_HEIGHT * 2,
            self.PADDLE_HALF_HEIGHT * 2 + self.HEIGHT,
            self.paddle_centers)

    def pong_step(self, paddle_direction):
        assert (np.all(np.abs(paddle_direction) <= 1))

        self.move_paddles(paddle_direction * self.PADDLE_STEP)
        self.move_ball()
        self.reflect_on_borders()
        self.reflect_on_paddle()

        self.update_game_state()


# Supports OpenAI Universe's vectorized interface
class BothPlayerPongGymEnv(gym.Env):
    metadata = {'render.modes': ['text_block', 'rgb_array', 'human']}

    def __init__(self, pong_game: PongGame = PongGame()):
        super().__init__()

        self.pong_game = pong_game

        self.action_space = gym.spaces.MultiDiscrete([(0, 3), (0, 3)])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(pong_game.HEIGHT, pong_game.HEIGHT))
        # We can keep the default reward_range.

        self.outputs = None

    def _step(self, actions):
        infos = ((),)
        # A bit awkward but I don't have an AI player, so I'll just make the agents
        # maximize the #steps in the game.
        # And play both agents.
        paddle_direction = -1 * (actions == GymAction.UP) + 1 * (actions == GymAction.DOWN)
        self.pong_game.pong_step(paddle_direction)

        rewards = np.array([1 if self.pong_game.game_state == GameState.PLAYING else 0])
        dones = np.array([self.pong_game.game_state != GameState.PLAYING])

        self.pong_game.handle_game_end()
        self.outputs = self.pong_game.render_pong_states()

        return self.outputs, rewards, dones, infos

    def _reset(self):
        self.pong_game.reset_game()
        self.outputs = self.pong_game.render_pong_states()

        return self.pong_game.render_pong_states()

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
