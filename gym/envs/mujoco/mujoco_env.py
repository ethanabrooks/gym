import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
# import six

import mujoco

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, action_space=None,
            observation_space=None):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__),
                                    "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.sim = mujoco.Sim(fullpath)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.joint_qpos.ravel().copy()
        self.init_qvel = self.sim.joint_qvel.ravel().copy()
        if action_space is None:
            bounds = self.sim.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low, high)
        else:
            self.action_space = action_space

        if observation_space is None:
            observation, _reward, done, _info = self._step(np.zeros(self.sim.nu))
            assert not done
            self.obs_dim = observation.size

            high = np.inf*np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:
            self.observation_space = observation_space
            

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every
        reset Optionally implement this method, if you need to tinker with
        camera position and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.sim.nq,) and qvel.shape == (self.sim.nv,)
        self.sim.joint_qpos[:] = qpos
        self.sim.joint_qvel[:] = qvel
        self.sim.forward()

    @property
    def dt(self):
        return self.sim.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.actuator_ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _render(self, *args, **kwargs):
        self.sim.render()

    def _get_viewer(self):
        return self.sim

    def get_body_com(self, body_name):
        return self.sims.get_xpos(mujoco.Types.BODY, body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.joint_qpos.flat,
            self.sim.joint_qvel.flat
        ])
