# Fishing Boats Simulation in OpenAI-Gym using Reinforcement Learning & PPO Algorithm, GUI Via PyGame.

The goal is to develop a fishing boat simulation using reinforcement learning, featuring a Fastboat deploying crab cages
and a Motherboat collecting them. PyGame renders the environment, and PPO algorithm trains the agents. The
goal is to create an autopilot system for maximum operational efficiency


### Description

This environment is a simulation sea fishing optimization problem.
The following environment has discrete actions: either one of the cages is resurfaced or not.
There are two environment versions: discrete or continuous.
The starting point of MotherBoat and FastBoat is always at coordinates (WIDTH // 2, 0) [Top Center].

To see a heuristic landing, run:

```
python env_gen.py
```

### Action Space

There are four discrete actions/moves available: UP, DOWN, LEFT, RIGHT

### Observation Space

The state is a list of three vectors: 
  - List of the coordinates of the motherboat in `x` & `y`, its linear velocities in `x` & `y`, its angle.
  - List of the coordinates of the fastboat in `x` & `y`, its linear velocities in `x` & `y`, its angle.
  - List of n booleans that represent whether each cage is resurfaced or not among n cages.

### Rewards

After every step a reward is granted. The total reward of an episode is the
sum of the rewards for all the steps within that episode.

For each step, the reward:
- is increased/decreased the closer/further the fastboat is to each one of the cages.
- is increased/decreased the faster/slower the fastboat is heading toward to each one of the cages.
- is increased by 10 points for each cage that is in contact with the fastboat.
- is increased by 10 points for each cage that is in collected by the motherboat.
- is decreased by 0.3 points each frame the motherboat engine is surfing.
- is decreased by 0.03 points each frame the fastboat is surfing.

The episode receive an additional reward of +100 * k/n points for resurfacing k cages [k: ranges from 0 to n cages].
An episode is considered a solution if it scores at least 100 points.

### Starting State

The FastBoat starts at the top center of the viewport with a random initial force applied to its center of mass.

### Episode Termination

The episode finishes if:

1. the fastboat crashes with some other boats (the fastboat body gets in contact with the some random boat);
2. the fastboat gets outside of the viewport (`x` coordinate is greater than 1);
3. the fastboat is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
    a body which is not awake is a body which doesn't move and doesn't
    collide with any other body:
> When Box2D determines that a body (or group of bodies) has come to rest,
> the body enters a sleep state which has very little CPU overhead. If a
> body is awake and collides with a sleeping body, then the sleeping body
> wakes up. Bodies will also wake up if a joint or contact attached to
> them is destroyed.

### Arguments

To use to the _continuous_ environment, you need to specify the `continuous=True` argument like below:

```python
import gym
env = gym.make(
    "SeaFisher-v1",
    continuous: bool = False,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
)
```
If `continuous=True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
The first coordinate of an action determines the throttle of the main engine, while the second
coordinate specifies the throttle of the lateral boosters.
Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
`main < 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
main engine doesn't work  with less than 50% power).
Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

`gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

If `enable_wind=True` is passed, there will be wind effects applied to the [FastBoat, MotherBoat, All Cages].
The wind is generated using the function `tanh(sin(2*k*(t+C)) + sin(pi*k*(t+C)))`.
`k` is set to 0.01.
`C` is sampled randomly between -9999 and 9999.

`wind_power` dictates the maximum magnitude of linear wind applied to the boats. The recommended value for `wind_power` is between 0.0 and 20.0.
`turbulence_power` dictates the maximum magnitude of rotational wind applied to the boats. The recommended value for `turbulence_power` is between 0.0 and 2.0.

### Version History

- v1: Initial version