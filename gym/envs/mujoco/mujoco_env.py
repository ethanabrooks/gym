import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
# import six

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install \
            mujoco_py, and also perform the setup instructions here: \
            https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_WIDTH = 255
DEFAULT_HEIGHT = 255


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip,
                 action_space=None, observation_space=None, offscreen=False):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__),
                                    "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.offscreen = offscreen

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        if action_space is None:
            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low, high)
        else:
            self.action_space = action_space

        if observation_space is None:
            action = self.action_space.sample()
            observation, _reward, done, _info = self._step(action)
            assert not done
            self.obs_dim = observation.size
            high = np.inf * np.ones(self.obs_dim)
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
        assert qpos.shape == (self.model.nq,) \
            and qvel.shape == (self.model.nv,)
        current = self.sim.get_state()
        new_state = mujoco_py.MjSimState(current.time, qpos, qvel,
                                         current.act, current.udd_state)
        self.sim.set_state(new_state)

        # TODO: what are these all about?
        # self.model._compute_subtree()  # pylint: disable=W0212
        # self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _render(self, mode='human', height=None, width=None, camera_name=None, close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer()  # .finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            assert None not in [height, width], 'You must specify dimensions \ 
                    for "rgb_array" mode'
            # raise RuntimeError('Use `_render_array`')
            return self.sim.render(height, width, camera_name=camera_name)
        elif mode == 'human':
            assert height is width is None, 'dimensions are set based on \
                    window size in "human" mode'
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
