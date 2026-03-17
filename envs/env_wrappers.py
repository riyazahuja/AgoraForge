"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import multiprocessing as mp
from abc import ABC, abstractmethod


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == 'snapshot':
            remote.send(env.snapshot())
        elif cmd == 'describe_actions':
            remote.send(env.describe_actions(data))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_env_snapshots(self):
        for remote in self.remotes:
            remote.send(('snapshot', None))
        return [remote.recv() for remote in self.remotes]

    def describe_actions(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('describe_actions', action))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
