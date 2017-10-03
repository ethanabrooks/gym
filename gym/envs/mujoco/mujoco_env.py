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

    def __init__(self, model_path, frame_skip):
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

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

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
            self.viewer.autoscale()
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

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer()  # .finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            raise RuntimeError('Use `_render_array`')
        elif mode == 'human':
            self._get_viewer().render()

    def _render_array(self, *args, **kwargs):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.
        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        return self.sim.render(*args, **kwargs)

    def _get_viewer(self):
        if self.viewer is None:
            # self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer = mujoco_py.MjRenderContextWindow(self.sim)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
