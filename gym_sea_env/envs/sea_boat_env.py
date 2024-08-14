import math
import random
import time
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

from gym_sea_env.envs.boat import Boat
from gym_sea_env.envs.cage import Cage

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    # Therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e


STATE_W = 96  # Less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Window Scale
PLAYFIELD = 1000 / SCALE  # Game Over Boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

# Colors
MOTHERBOAT_COLOR = (231, 76, 60)
FASTBOAT_COLOR = (39, 174, 96)
BLOCKBOAT_COLOR = (52, 73, 94)

# Environment

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

CAGES_DIM = PLAYFIELD / 20.0

class CagesDetector(contactListener):
    """
        Start CagesDetection:
        - Happens to touch a cage of cages so contact is maintained
        - Happens to touch the motherboat once a mission of activation is finished so contact is maintained
    """
    def __init__(self, env, fishing_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.fishing_complete_percent = fishing_complete_percent

    def BeginContact(self, contact):
        self._contact(contact)

    def EndContact(self, contact):
        pass
        # self._contact(contact)

    def _contact(self, contact):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        objectA = fixtureA.body.userData
        objectB = fixtureB.body.userData

        if (isinstance(objectA, str) and isinstance(objectB, str)):
            if ('fastboat' in objectA and 'cage' in objectB) or ('cage' in objectA and 'fastboat' in objectB):
                print("Visited Cages Cnt: ", self.env.cage_visited_count)
                cage_index = objectB if 'fastboat' in objectA else objectA
                cage = self.env.cages[cage_index]
                self.env.cage_visited_count += bool(not cage.is_active)
                if not cage.is_active:
                    self.env.fastboat.brake(0)
                cage.activate()
                self.env.reward += 1500.0 / len(self.env.cages)
                # Fishing is considered completed all of the cages were captured
                if self.env.cage_visited_count == len(self.env.cages):
                    self.env.reward += 1000.0
                    self.env.game_over = True
                # print("Contact between Fastboat and Cage detected!")
            elif ('fastboat' in objectA and 'blockboat' in objectB) or ('blockboat' in objectA and 'fastboat' in objectB):
                self.env.game_over = True
                # print("Contact between Fastboat and Blockboat detected!")
            elif ('fastboat' in objectA and 'motherboat' in objectB) or ('motherboat' in objectA and 'fastboat' in objectB):
                # Fishing is considered pseudo-completed if enough % of the cages were captured
                # This means the motherboat can capture the fastboat
                if (self.env.cage_visited_count / len(self.env.cages) > self.fishing_complete_percent):
                    self.env.game_over = True
                # print("Contact between Fastboat and Motherboat detected!")
            elif ('cage' in objectA and 'motherboat' in objectB) or ('motherboat' in objectA and 'cage' in objectB):
                # Only if the cage is active it should be removed provide me the code
                cage_name = objectB if 'motherboat' in objectA else objectA
                cage = self.env.cages[cage_name]
                if cage.is_active:
                    cage.destroy()
                    self.env.cages.pop(cage_name)

                # self.env.game_over = True
                # print("Contact between Cage and Motherboat detected!")
            elif ('blockboat' in objectA and 'motherboat' in objectB) or ('motherboat' in objectA and 'blockboat' in objectB):
                pass
                # Execute your desired code here
                # This block will be executed when there is contact between a Blockboat and a Motherboat
                # print("Contact between Blockboat and Motherboat detected!")


class SeaBoatEnv(gym.Env, EzPickle):
    """
    ### Description
    The easiest control task to learn from pixels - a top-down
    fishing environment. The generated objects are random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym_sea_env/envs/sea_boat_env.py
    ```
    Remember: it's a powerful rear-outboarding engine boat - don't press the accelerator
    and turn at the same time.

    ### Continuous Action Space
    The continuous action space has 2 actions: [steer, throttle].
    The steer action is a single float in [-1, 1] which specifies the steering angle of the front wheels.
    The throttle action is a single float in [0, 1] which specifies the amount of throttle to apply.

    ### Discrete Action Space
    The discrete action space has 4 actions: [do nothing, left, right, throttle].
    The left and right actions are binary, and turn the front wheels 0.6 radians left or right respectively.
    The throttle and brake actions are binary, and add or remove 0.2 of throttle from the fastboat respectively.
    The do nothing action does nothing.

    ### Observation Space
    State consists of 96x96 pixels.

    ### Rewards
    The reward is -0.1 every frame and +1000/N for every fishing cage visited,
    where N is the total number of cages visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ### Starting State
    The fastboat starts at rest in some random position in the sea.

    ### Episode Termination
    The episode finishes when all of the cages are visited. The fastboat can also go
    outside of the playfield - that is, far off the playfield delimited, in which case it will
    receive -100 reward and die.

    ### Create the Boats and Cages
    The boats and cages are created in the `_create_fastboat`, `_create_motherboat`, `_create_blockboats` and `_create_cages` methods.
    The `_create_fastboat` method creates the fastboat.
    The `_create_motherboat` method creates the motherboat.
    The `_create_blockboats` method creates the blockboats.
    The `_create_cages` method creates the cages.

    ### Create the sea
    The sea is created in the as an environment background color.

    ### Arguments
    `fishing_complete_percent` dictates the percentage of cages that must be visited by
    the agent before a fishing frame is considered complete.

    `render_mode` dictates the mode of rendering the environment. The options are:
    - `human`: renders the environment in a window on the screen.
    - `rgb_array`: returns the RGB array of the rendered frame.
    - `state_pixels`: returns the state pixels of the rendered frame.

    `continuous` dictates the action space of the environment. The options are:
    - `True`: the action space is continuous.
    - `False`: the action space is discrete.

    `verbose` dictates whether to print out information about the environment. The options are:
    - `True`: print out information about the environment.
    - `False`: do not print out information about the environment.

    ### Example Usage
    ```py
    import gymnasium as gym
    import gym_sea_env

    env = gym.make("CarRacing-v1", render_mode="human", continuous=True, verbose=True)

    # Normal reset,
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        fishing_complete_percent: float = 0.85,
        render_mode: Optional[str] = 'human',
        continuous: bool = True,
        verbose: bool = False,
    ):
        EzPickle.__init__(
            self,
            fishing_complete_percent,
            render_mode,
            continuous,
            verbose,
        )
        self.continuous = continuous
        self.fishing_complete_percent = fishing_complete_percent
        self.contactListener_keepref = CagesDetector(self, self.fishing_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.cages: Optional[Dict[Cage]] = dict()
        self.fastboat: Optional[Boat] = None
        self.motherboat: Optional[Boat] = None
        self.blockboats: Optional[Dict[Boat]] = dict()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.game_over = False
        self.sea_color = (135, 206, 235)
        self.sea_fixtureDef = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1.0, -0.2]).astype(np.float32),
                np.array([+1.0, +1.0]).astype(np.float32),
            )  # Steer, Throttle
        else:
            self.action_space = spaces.Discrete(4)
            # Nothing, Left, Right, Throttle

        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        self.render_mode = render_mode

    def _destroy(self):
        if len(self.cages) == 0:
            return
        self.world.contactListener = None
        for cage in self.cages.values():
            cage.destroy()
        self.cages.clear()
        
        assert self.fastboat is not None
        self.fastboat.destroy()
        assert self.motherboat is not None
        self.motherboat.destroy()
        for blockboat in self.blockboats.values():
            blockboat.destroy()
        self.blockboats.clear()

    def _random_angle(self):
        return random.uniform(-math.pi, +math.pi)

    def _random_coords(self):
        return (random.uniform(-PLAYFIELD, +PLAYFIELD), random.uniform(-PLAYFIELD, +PLAYFIELD))

    def _random_position(self):
        objects_list = []
        objects_list.extend(self.cages.values())
        objects_list.extend(self.blockboats.values())
        objects_list.append(self.motherboat)
        objects_list.append(self.fastboat)
        filled_positions = []
        for c in objects_list:
            if c is not None:
                try: filled_positions.append(c.hull.position)
                except: pass

        while True:
            x, y = self._random_coords()
            if all([all([abs(x - i) < 20, abs(y - j) < 20])] for i, j in filled_positions):
                break
            continue
        return x, y

    def _create_motherboat(self, ):
        x, y = self._random_position()
        self.motherboat = Boat(self.world, 'motherboat', MOTHERBOAT_COLOR, 1.5, self._random_angle(), x, y)

    def _create_fastboat(self, ):
        self.fastboat = Boat(self.world, 'fastboat', FASTBOAT_COLOR, 0.9, self._random_angle(), 0, 0, 1.6)

    def _create_blockboats(self, num_blockboats=10):
        for i in range(num_blockboats):
            x, y = self._random_position()
            name = f'blockboat_{i}'
            blockboat = Boat(self.world, name, BLOCKBOAT_COLOR, 1.1, self._random_angle(), x, y)
            self.blockboats[name] = blockboat

    def _create_cages(self, num_cages=10):
        for i in range(num_cages):
            init_position = self._random_position()
            name = f'cage_{i}'
            cage = Cage(self.world, name, init_position)
            self.cages[name] = cage

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = CagesDetector(self, self.fishing_complete_percent)
        self.world.contactListener = self.world.contactListener_keepref
        self.reward = 0.0
        self.game_over = False
        self.prev_reward = 0.0
        self.cage_visited_count = 0
        self.t = 0.0
        
        self._create_fastboat()
        self._create_motherboat()
        self._create_blockboats(num_blockboats=5)
        self._create_cages(400)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        assert self.fastboat is not None, "You forgot to call reset()"
        assert self.motherboat is not None, "You forgot to call reset()"
        assert not self.game_over, "Cannot call step() after game_over"

        if action is not None:
            if self.continuous:
                # Fastboat action
                self.fastboat.steer(-action[0])
                self.fastboat.throttle(action[1])
                self.fastboat.brake(np.random.uniform(0, 1))
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.fastboat.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.fastboat.throttle(0.2 * (action == 3))
                self.fastboat.brake(0.8 * (action == 4))

        motherboat_action = np.random.uniform(-1, 1, size=3)
        if motherboat_action is not None:
            if self.continuous: # Motherboat action
                self.motherboat.steer(-motherboat_action[0])
                self.motherboat.throttle(motherboat_action[1])
                self.motherboat.brake(motherboat_action[2])
        
        # Blockboats action
        for blockboat in self.blockboats.values():
            blockboat_action = np.random.uniform(-1, 1, size=3)
            if blockboat_action is not None:
                if self.continuous: # Blockboat action
                    blockboat.steer(-blockboat_action[0])
                    blockboat.throttle(blockboat_action[1])
                    blockboat.brake(blockboat_action[2])

        self.fastboat.step(1.0 / FPS)
        self.motherboat.step(1.0 / FPS)
        for blockboat in self.blockboats.values():
            blockboat.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.state = self._render("state_pixels")

        step_reward = 0.0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want boat to be faster.
            # self.reward -=  10 * self.fastboat.fuel_spent / ENGINE_POWER
            self.fastboat.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if (self.cage_visited_count / len(self.cages) > self.fishing_complete_percent):
                # This doesn't refer to some failure but the end of the fishing is almost complete
                truncated = True
            x, y = self.fastboat.hull.position
            if self.game_over or abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

            x, y = self.motherboat.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
            
            for name in list(self.blockboats.keys()):
                blockboat = self.blockboats[name]
                x, y = blockboat.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    blockboat.destroy()
                    self.blockboats.pop(name)

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def render(self):
        return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]
        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        assert self.fastboat is not None
        
        # Computing Transformations
        angle = -self.fastboat.hull.angle
        
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.fastboat.hull.position[0]) * zoom
        scroll_y = -(self.fastboat.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_world(zoom, trans, angle)
        # Draw Fastboat
        self.fastboat.draw(self.surf, zoom, trans, angle, mode not in ["state_pixels_list", "state_pixels"])
        # Draw Motherboat
        self.motherboat.draw(self.surf, zoom, trans, angle, mode not in ["state_pixels_list", "state_pixels"])
        self.surf = pygame.transform.flip(self.surf, False, True)

        # Showing stats
        # self._render_indicators(WINDOW_W, WINDOW_H)

        # Reward
        font = pygame.font.Font(pygame.font.get_default_font(), 22)
        reward_text = font.render("Reward: %04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        reward_text_rect = reward_text.get_rect()
        reward_text_rect.center = (80, WINDOW_H - WINDOW_H * 1.4 / 40.0)
        self.surf.blit(reward_text, reward_text_rect)

        # Cages visited
        cage_text = font.render("Cages visited: %03i" % self.cage_visited_count, True, (255, 255, 255), (0, 0, 0))
        cage_text_rect = cage_text.get_rect()
        cage_text_rect.center = (WINDOW_W - 200, WINDOW_H - WINDOW_H * 1.2 / 40.0)
        self.surf.blit(cage_text, cage_text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_world(self, zoom, translation, angle):
        PLAYFIELD_RECT = [
            (PLAYFIELD, PLAYFIELD),
            (PLAYFIELD, -PLAYFIELD),
            (-PLAYFIELD, -PLAYFIELD),
            (-PLAYFIELD, PLAYFIELD),
        ]
        
        # Draw Sea background
        self._draw_colored_polygon(self.surf, PLAYFIELD_RECT, self.sea_color, zoom, translation, angle, clip=False)

        # Draw Cages
        for cage in self.cages.values():
            cage.draw(self.surf, zoom, translation, angle)
        
        # Draw Blockboats
        for blockboat in self.blockboats.values():
            blockboat.draw(self.surf, zoom, translation, angle)

    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle, clip=True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    print("UPPPPP")
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = SeaBoatEnv(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
