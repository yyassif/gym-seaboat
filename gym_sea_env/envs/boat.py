import math

import Box2D
import numpy as np
from gymnasium.error import DependencyNotInstalled

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

SIZE = 0.02
ENGINE_POWER = 300000000 * SIZE * SIZE
FRICTION_LIMIT = 1000000 * SIZE * SIZE # friction ~= mass ~= size^2 (calculated implicitly using density)
ENGINE_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
HULL_POLY = [(0, 140), (20, 100), (30.0, -50.0), (20, -100), (-20, -100), (-30.0, -50.0), (-20, 100)]
ENGINE_POS = (0, -120)
ENGINE_COLOR = (0, 0, 0)
ENGINE_WHITE = (77, 77, 77)
WHITE_COLOR = (255, 255, 255)
ENGINE_R = 20

class Boat:
    def __init__(self, world, user_data, color, size_scale, init_angle, init_x, init_y, speed_ratio=1.0):
        self.world: Box2D.b2World = world
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            userData=user_data,
            position=Box2D.b2Vec2((init_x, init_y)),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE * size_scale, y * SIZE * size_scale) for x, y in HULL_POLY]
                    ),
                    density=1.0,
                )
            ],
        )
        self.hull.color = color
        self.fuel_spent = 0.0
        ENGINE_POLY = [
            (-ENGINE_R, +ENGINE_R),
            (+ENGINE_R, +ENGINE_R),
            (+ENGINE_R, -ENGINE_R),
            (-ENGINE_R, -ENGINE_R),
        ]
        self.outboard_engine = self.world.CreateDynamicBody(
                position=(init_x + ENGINE_POS[0] * SIZE * size_scale, init_y + ENGINE_POS[1] * SIZE * size_scale),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * SIZE * size_scale, y * SIZE * size_scale) for x, y in ENGINE_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
        self.outboard_engine.wheel_rad = ENGINE_R * SIZE
        self.outboard_engine.color = ENGINE_COLOR
        self.outboard_engine.throttle = 0.0
        self.outboard_engine.brake = 0.0
        self.outboard_engine.steer = 0.0
        self.outboard_engine.phase = 0.0  # Wheel angle
        self.outboard_engine.omega = 0.0  # Angular velocity
        self.outboard_engine.trail_start = None
        self.outboard_engine.trail_particle = None
        rjd = revoluteJointDef(
            bodyA=self.hull,
            bodyB=self.outboard_engine,
            localAnchorA=(ENGINE_POS[0] * SIZE * size_scale, ENGINE_POS[1] * SIZE * size_scale),
            localAnchorB=(0, 0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=180 * 900 * SIZE * SIZE,
            motorSpeed=0,
            lowerAngle=-0.53,
            upperAngle=+0.53,
        )
        self.outboard_engine.joint = self.world.CreateJoint(rjd)
        self.outboard_engine.cages = set()
        self.outboard_engine.userData = self.outboard_engine
        self.drawlist = [self.outboard_engine, self.hull]
        self.speed_ratio = speed_ratio
        self.particles = []

    def throttle(self, throttle):
        """Control: Rear wheel drive
        Args:
            throttle (float): How much throttle gets applied. Gets clipped between 0 and 1.
        """
        throttle = np.clip(throttle, 0, 1)
        self.outboard_engine
        diff = throttle - self.outboard_engine.throttle
        if diff > 0.1:
            diff = 0.1  # Gradually increase, But stop immediately
        self.outboard_engine.throttle += diff

    def brake(self, b):
        """control: Brake
        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        self.outboard_engine.brake = b

    def steer(self, s):
        """control: steer
        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.outboard_engine.steer = s

    def step(self, dt):
        # Steer the outboard engine
        dir = np.sign(self.outboard_engine.steer - self.outboard_engine.joint.angle)
        val = abs(self.outboard_engine.steer - self.outboard_engine.joint.angle)
        self.outboard_engine.joint.motorSpeed = dir * min(50.0 * val, 3.0)

        # Position => Friction_limit
        friction_limit = FRICTION_LIMIT * 0.7  # Sea friction

        # Force
        forw = self.outboard_engine.GetWorldVector((0, 1))
        side = self.outboard_engine.GetWorldVector((1, 0))
        v = self.outboard_engine.linearVelocity
        vf = forw[0] * v[0] + forw[1] * v[1]  # Forward speed
        vs = side[0] * v[0] + side[1] * v[1]  # Side speed

        # ENGINE_MOMENT_OF_INERTIA*np.square(self.outboard_engine.omega)/2 = E -- energy
        # ENGINE_MOMENT_OF_INERTIA*self.outboard_engine.omega * domega/dt = dE/dt = W -- power
        # domega = dt*W/ENGINE_MOMENT_OF_INERTIA/self.outboard_engine.omega

        # Add small coefficient not to divide by Zero
        self.outboard_engine.omega += (
            dt
            * self.speed_ratio
            * ENGINE_POWER
            * self.outboard_engine.throttle
            / ENGINE_MOMENT_OF_INERTIA
            / (abs(self.outboard_engine.omega) + 5.0)
        )
        self.fuel_spent += dt * ENGINE_POWER * self.outboard_engine.throttle

        # Braking for energy and power consumption in the engine
        if self.outboard_engine.brake >= 0.9:
            self.outboard_engine.omega = 0
        elif self.outboard_engine.brake > 0:
            BRAKE_FORCE = 15  # Radians per second
            dir = -np.sign(self.outboard_engine.omega)
            val = BRAKE_FORCE * self.outboard_engine.brake
            if abs(val) > abs(self.outboard_engine.omega):
                val = abs(self.outboard_engine.omega)  # Low speed => same as = 0
            self.outboard_engine.omega += dir * val
        self.outboard_engine.phase += self.outboard_engine.omega * dt

        vr = self.outboard_engine.omega * self.outboard_engine.wheel_rad  # Rotating wheel speed
        f_force = -vf + vr  # Force direction is direction of speed difference
        p_force = -vs

        # Random coefficient to cut oscillations in few steps
        f_force *= 205000 * SIZE * SIZE
        p_force *= 205000 * SIZE * SIZE
        
        force = np.sqrt(np.square(f_force) + np.square(p_force))

        # Trail trace
        if abs(force) > 2.0 * friction_limit:
            if (self.outboard_engine.trail_particle and len(self.outboard_engine.trail_particle.poly) < 30):
                self.outboard_engine.trail_particle.poly.append((self.outboard_engine.position[0], self.outboard_engine.position[1]))
            elif self.outboard_engine.trail_start is None:
                self.outboard_engine.trail_start = self.outboard_engine.position
            else:
                self.outboard_engine.trail_particle = self._create_particle(self.outboard_engine.trail_start, self.outboard_engine.position)
                self.outboard_engine.trail_start = None
        else:
            self.outboard_engine.trail_start = None
            self.outboard_engine.trail_particle = None

        if abs(force) > friction_limit:
            f_force /= force
            p_force /= force
            force = friction_limit  # Correct physics here
            f_force *= force
            p_force *= force

        self.outboard_engine.omega -= dt * f_force * self.outboard_engine.wheel_rad / ENGINE_MOMENT_OF_INERTIA

        self.outboard_engine.ApplyForceToCenter(
            (
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1],
            ),
            True,
        )

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in poly
                ]
                pygame.draw.lines(surface, color=p.color, points=poly, width=2, closed=False)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in path
                ]

                pygame.draw.polygon(surface, color=obj.color, points=path)

                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # Radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-ENGINE_R * SIZE, +ENGINE_R * c1 * SIZE),
                    (+ENGINE_R * SIZE, +ENGINE_R * c1 * SIZE),
                    (+ENGINE_R * SIZE, +ENGINE_R * c2 * SIZE),
                    (-ENGINE_R * SIZE, +ENGINE_R * c2 * SIZE),
                ]
                white_poly = [trans * v for v in white_poly]
                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly]
                white_poly = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=ENGINE_WHITE, points=white_poly)

    def _create_particle(self, point1, point2):
        class Particle:
            pass

        p = Particle()
        p.color = WHITE_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        self.world.DestroyBody(self.outboard_engine)
        self.outboard_engine = None
        

