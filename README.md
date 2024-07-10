# An-illstrate-of-customed-gym-env
Hi,

I'm a student and I'm trying to use MuJoCo for simulating a DVRK robot and implement Reinforcement Learning as control.

I'm looking for some help with a error I encountered while creating a custom gym environment.

Here is a code of the custom gym environment which explains my question:

`import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

class DvrkEnv(MujocoEnv, utils.EzPickle):
"""
### Description
"DVRK" is a multi-jointed surgical robot. The goal is to move the robot's end effector (called psm_gripper1_link) close to a
target that is spawned at a random position.


### Action Space
The action space is a `Box(-1, 1, (8,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
| 0   |  Torque applied at psm_yaw_link                      | -1 | 1 | psm_yaw_link  | hinge | torque (N m) |
| 1   |  Torque applied psm_pitch_back_link                  | -1 | 1 | psm_pitch_back_link  | hinge | torque (N m) |
| 2   |  Torque applied psm_main_insertion_link              | -1 | 1 | psm_main_insertion_link | slide | torque (N m) |
| 3   |  Torque applied psm_tool_roll                        | -1 | 1 | psm_ | slide | torque (N m) |
| 4   |  Torque applied psm_tool_pitch                       | -1 | 1 | psm_main_insertion_link | slide | torque (N m) |
| 5   |  Torque applied psm_tool_yaw                         | -1 | 1 | psm_main_insertion_link | slide | torque (N m) |
| 6   |  Torque applied psm_gripper1                         | -1 | 1 | psm_main_insertion_link | slide | torque (N m) |
| 6   |  Torque applied psm_gripper2                         | -1 | 1 | psm_main_insertion_link | slide | torque (N m) |


### Observation Space

Observations consist of

The observation is a `ndarray` with shape `(21,)` where the elements correspond to the following:

| Num | Observation                                                                               | Min  | Max | Name (in corresponding XML file)      | Joint | Unit                     |
| --- | ------------------------------------------------------------------------------------------| ---- | --- | --------------------------------      | ----- | ------------------------ |
| 0   | Angle of the psm_yaw_joint                                                                | -Inf | Inf | psm_yaw_joint                         | hinge | unitless                 |
| 1   | Angle of the psm_pitch_back_link                                                          | -Inf | Inf | psm_pitch_back_joint                  | hinge | unitless                 |
| 2   | Distance moved by main_insertion link                                                     | -Inf | Inf | psm_pitch_back_joint                  | slide | unitless                 |
| 3   | Angle of the psm_tool_roll_link                                                           | -Inf | Inf | psm_tool_roll_joint                   | hinge | unitless                 |
| 4   | Angle of the psm_tool__pitch_link                                                         | -Inf | Inf | psm_tool_pitch_joint                  | hinge | unitless                 |
| 5   | Angle of the psm_tool_yaw_link                                                            | -Inf | Inf | psm_tool_yaw_joint                    | hinge | unitless                 |
| 6   | Angle of the psm_tool_gripper1                                                            | -Inf | Inf | psm_tool_gripper1_joint               | hinge | unitless                 |
| 7   | Angle of the psm_tool_gripper2                                                            | -Inf | Inf | psm_tool_gripper2_joint               | hinge | unitless                 |
| 8   | x-coordinate of the target                                                                | -Inf | Inf | target_x                              | slide | position (m)             |
| 9   | y-coordinate of the target                                                                | -Inf | Inf | target_y                              | slide | position (m)             |
| 10  | z-coordinate of the target                                                                | -Inf | Inf | target_y                              | slide | position (m)             |

| 11  | Angular velocity psm_yaw_joint                                                            | -Inf | Inf | psm_yaw_joint                         | hinge | angular velocity (rad/s) |
| 12  | Angular velocity of the psm_pitch_back_link                                               | -Inf | Inf | psm_pitch_back_joint                  | hinge | unitless                 |
| 13  | Speed main_insertion link                                                                 | -Inf | Inf | psm_pitch_back_joint                  | slide | unitless                 |
| 14  | Angular velocity of the psm_tool_roll_link                                                |-Inf | Inf | psm_tool_roll_joint                    | hinge | unitless                 |
| 15  | Angular velocity of the psm_tool__pitch_link                                              | -Inf | Inf | psm_tool_pitch_joint                  | hinge | unitless                 |
| 16  | Angular velocity of the psm_tool_yaw_link                                                 | -Inf | Inf | psm_tool_yaw_joint                    | hinge | unitless                 |
| 17  | Angular velocity of the psm_tool_gripper1                                                 | -Inf | Inf | psm_tool_gripper1_joint               | hinge | unitless                 |
| 18  | Angular velocity of the psm_tool_gripper2                                                 | -Inf | Inf | psm_tool_gripper2_joint               | hinge | unitless                 |

| 19  | position_gripper1- position_target                                             | -Inf | Inf | NA                                               | slide | position (m)             |



### Rewards
The reward consists of two parts:
- *reward_distance*: This reward is a measure of how far the *psm_tool_gripper1_link*
of the reacher (the unattached end) is from the target, with a more negative
value assigned for when the dvrk's *psm_tool_gripper1* is further away from the
target. It is calculated as the negative vector norm of (position of
the gripper1 - position of target), or *-norm("psm_tool_gripper1_link" - "target")*.
- *reward_control*: A negative reward for penalising the walker if
it takes actions that are too large. It is measured as the negative squared
Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

The total reward returned is ***reward*** *=* *reward_distance + reward_control*


### Starting State
All observations start in state
(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
with a noise added for stochasticity. A uniform noise in the range
[-0.1, 0.1] is added to the positional attributes, while the target position
is selected uniformly at random in a disk of radius 0.2 around the origin.
Independent, uniform noise in the
range of [-0.005, 0.005] is added to the velocities, and the last
element ("fingertip" - "target") is calculated at the end once everything
is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*

### Episode End

The episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 50 timesteps (with a new random target popping up if the dvrk's psm_tool_fripper1_link reaches it before 50 timesteps)
2. Termination: Any of the state space values is no longer finite.

### Arguments

No additional arguments are currently supported (in v2 and lower),
but modifications can be made to the XML file in the assets folder
(or by changing the path to a modified XML file in another folder)..

```
env = gym.make('Dvrk')
```
"""

metadata = {
    "render_modes": [
        "human",
        "rgb_array",
        "depth_array",
    ],
    "render_fps": 50,
}

def __init__(self, **kwargs):
    utils.EzPickle.__init__(self, **kwargs)
    observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64)
    MujocoEnv.__init__(
        self, "dvrk.xml", 2, observation_space=observation_space,  render_mode = 'human', **kwargs
    )

def step(self, a):
    vec = self.get_body_com("psm_tool_gripper1_link") - self.get_body_com("target")
    reward_dist = -np.linalg.norm(vec)
    reward_ctrl = -np.square(a).sum()
    reward = reward_dist + reward_ctrl

    self.do_simulation(a, self.frame_skip)
    if self.render_mode == "human":
        self.render()

    ob = self._get_obs()
    return (
        ob,
        reward,
        False,
        False,
        dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
    )

def viewer_setup(self):
    assert self.viewer is not None
    self.viewer.cam.trackbodyid = 0

def reset_model(self):
    qpos = (
        self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        + self.init_qpos
    )
    while True:
        self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=3)
        if np.linalg.norm(self.goal) < 0.2:
            break
    qpos[-3:] = self.goal
    qvel = self.init_qvel + self.np_random.uniform(
        low=-0.005, high=0.005, size=self.model.nv
    )
    qvel[-3:] = 0
    self.set_state(qpos, qvel)
    return self._get_obs()

def _get_obs(self):
    return np.concatenate(
        [
            self.data.qpos[1],
            self.data.qpos[2],
            self.data.qpos[5],
            self.data.qpos[6],
            self.data.qpos[7],
            self.data.qpos[8],
            self.data.qpos[9],
            self.data.qpos[10],
            self.data.qpos[13],
            self.data.qpos[14],
            self.data.qpos[15],

            self.data.qvel[1],
            self.data.qvel[2],
            self.data.qvel[5],
            self.data.qvel[6],
            self.data.qvel[7],
            self.data.qvel[8],
            self.data.qvel[9],
            self.data.qvel[10],

            self.get_body_com("psm_tool_gripper1_link") - self.get_body_com("target"),
        ]
    )
