"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.
    resid_pdrop = 0.
    attn_pdrop = 0.

    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_pdrop = config.attn_pdrop

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        head_dim = C // self.n_head
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_head, head_dim)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        dropout_p = self.attn_pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None), dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, new_kv_cache


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            GELU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache


class GPT(nn.Module):
    def __init__(self, config, model_type='actor'):
        super().__init__()

        self.config = config
        self.model_type = config.model_type
        self.state_size = config.state_size

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        if model_type == 'actor':
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        elif model_type == 'critic':
            self.head = nn.Linear(config.n_embd, 1, bias=False)
        else:
            raise NotImplementedError

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, config.n_embd), nn.Tanh())
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.mask_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)
        return optimizer

    def forward(self, states, pre_actions, rtgs=None, timesteps=None, kv_caches=None):
        state_embeddings = self.state_encoder(
            states.reshape(-1, self.state_size).type(torch.float32).contiguous())
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)

        if self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = state_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb + context_pos_emb

        x = self.drop(token_embeddings + position_embeddings)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv_cache=layer_cache)
            new_kv_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        if self.model_type == 'rtgs_state_action':
            logits = logits[:, 2::3, :]
        elif self.model_type == 'state_action':
            logits = logits[:, 1::2, :]
        elif self.model_type == 'state_only':
            logits = logits
        else:
            raise NotImplementedError()

        return logits, new_kv_caches
