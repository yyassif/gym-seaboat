import Box2D
from gymnasium.error import DependencyNotInstalled

try:
    from Box2D.b2 import fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

CAGE_COLOR = (152, 153, 155)
CAGE_ACTIVE_COLOR = (62, 188, 46)
CAGE_HULL_POLY =  [
                  [(0, 4), (1, 2), (0, 0), (-1, 2)],
                  [(3, 3), (2, 1), (0, 0), (1, 2)],
                  [(4, 0), (2, -1), (0, 0), (2, 1)],
                  [(3, -3), (1, -2), (0, 0), (2, -1)],
                  [(0, -4), (-1, -2), (0, 0), (1, -2)],
                  [(-3, -3), (-2, -1), (0, 0), (-1, -2)],
                  [(-4, 0), (-2, 1), (0, 0), (-2, -1)],
                  [(-3, 3), (-1, 2), (0, 0), (-2, 1)],
                  ]

class Cage:
  def __init__(self, world, user_data, init_pos):
    self.is_active = False
    self.world: Box2D.b2World = world
    self.hull: Box2D.b2Body = self.world.CreateStaticBody(
      userData=user_data,
      position=Box2D.b2Vec2(init_pos),
      fixtures=[
        fixtureDef(
          shape=polygonShape(
            vertices=[(i/4, j/4) for i, j in HULL_POLY]
          ),
          density=0.5,
          friction=0.1,
          restitution=0.1,
          categoryBits=0x0002,
          maskBits=0x0001,  # Collides with bodies in category 1
          isSensor=True,   # Treats the fixture as a sensor
        ) for HULL_POLY in CAGE_HULL_POLY
      ],
    )
    self.hull.color = CAGE_COLOR

  def activate(self):
    self.is_active = True
    self.hull.color = CAGE_ACTIVE_COLOR

  def draw(self, surface, zoom, translation, angle):
    import pygame.draw
    for f in self.hull.fixtures:
      trans = f.body.transform
      path = [trans * v for v in f.shape.vertices]
      path = [(coords[0], coords[1]) for coords in path]
      path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
      path = [(coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in path]
      pygame.draw.polygon(surface, color=self.hull.color, points=path)

  def destroy(self):
    try:
      self.world.DestroyBody(self.hull)
    except:
      pass
    finally:
      self.hull = None

