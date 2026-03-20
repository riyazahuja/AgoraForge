import copy
import random
import numpy as np
import torch
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_model_device(model):
    return next(unwrap_model(model).parameters()).device


def sample(model, critic_model, state, obs, sample=False, actions=None, rtgs=None,
           timesteps=None, available_actions=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time.
    """
    block_size = unwrap_model(model).get_block_size()
    model.eval()
    critic_model.eval()

    obs_cond = obs if obs.size(1) <= block_size//3 else obs[:, -block_size//3:]
    state_cond = state if state.size(1) <= block_size//3 else state[:, -block_size//3:]
    if actions is not None:
        actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:]
    rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:]
    timesteps = timesteps if timesteps.size(1) <= block_size//3 else timesteps[:, -block_size//3:]

    logits, _ = model(obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps)
    logits = logits[:, -1, :]
    if available_actions is not None:
        logits[available_actions == 0] = -1e10
    probs = F.softmax(logits, dim=-1)

    if sample:
        a = torch.multinomial(probs, num_samples=1)
    else:
        _, a = torch.topk(probs, k=1, dim=-1)

    v, _ = critic_model(state_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps)
    v = v.detach()
    v = v[:, -1, :]

    return a, v


def padding_obs(obs, target_dim):
    len_obs = np.shape(obs)[-1]
    if len_obs > target_dim:
        print("target_dim (%s) too small, obs dim is %s." % (target_dim, len_obs))
        raise NotImplementedError
    elif len_obs < target_dim:
        padding_size = target_dim - len_obs
        if isinstance(obs, list):
            obs = np.array(copy.deepcopy(obs))
            padding = np.zeros(padding_size)
            obs = np.concatenate((obs, padding), axis=-1).tolist()
        elif isinstance(obs, np.ndarray):
            obs = copy.deepcopy(obs)
            shape = np.shape(obs)
            padding = np.zeros((shape[0], shape[1], padding_size))
            obs = np.concatenate((obs, padding), axis=-1)
        else:
            print("unknown type %s." % type(obs))
            raise NotImplementedError
    return obs


def padding_ava(ava, target_dim):
    len_ava = np.shape(ava)[-1]
    if len_ava > target_dim:
        print("target_dim (%s) too small, ava dim is %s." % (target_dim, len_ava))
        raise NotImplementedError
    elif len_ava < target_dim:
        padding_size = target_dim - len_ava
        if isinstance(ava, list):
            ava = np.array(copy.deepcopy(ava), dtype=np.long)
            padding = np.zeros(padding_size, dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1).tolist()
        elif isinstance(ava, np.ndarray):
            ava = copy.deepcopy(ava)
            shape = np.shape(ava)
            padding = np.zeros((shape[0], shape[1], padding_size), dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1)
        else:
            print("unknown type %s." % type(ava))
            raise NotImplementedError
    return ava
