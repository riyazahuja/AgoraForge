"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import copy

import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import unwrap_model


class TrainerConfig:
    max_epochs = 1000
    batch_size = 128
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.5
    weight_decay = 0.1
    num_workers = 0
    mode = "offline"
    device = torch.device("cpu")
    distributed = False
    rank = 0
    world_size = 1
    use_distributed_sampler = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, critic_model, config):
        self.model = model
        self.critic_model = critic_model
        self.config = config

        self.raw_model = unwrap_model(self.model)
        self.raw_critic_model = unwrap_model(self.critic_model)
        self.device = getattr(config, "device", next(self.raw_model.parameters()).device)

        self.optimizer = self.raw_model.configure_optimizers(config, config.learning_rate)
        self.critic_optimizer = self.raw_critic_model.configure_optimizers(config, config.learning_rate * 3)

        self._use_amp = torch.cuda.is_available()
        self._amp_dtype = torch.bfloat16


    def _reduce_metrics(self, metric_sums, count):
        values = torch.tensor(
            [
                metric_sums["actor_loss"],
                metric_sums["critic_loss"],
                metric_sums["entropy"],
                metric_sums["ratio"],
                metric_sums["confidence"],
                float(count),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if self.config.distributed:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)

        total_count = max(values[-1].item(), 1.0)
        return (
            values[0].item() / total_count,
            values[1].item() / total_count,
            values[2].item() / total_count,
            values[3].item() / total_count,
            values[4].item() / total_count,
        )

    def _build_loader(self, dataset, epoch):
        if self.config.distributed and self.config.use_distributed_sampler:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True,
                drop_last=False,
            )
            sampler.set_epoch(epoch)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        batch_size = max(1, min(int(self.config.batch_size), len(dataset)))
        return DataLoader(
            dataset,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            batch_size=batch_size,
            num_workers=0,
        )

    def train(self, dataset, train_critic=True):
        if len(dataset) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        model, critic_model, config = self.model, self.critic_model, self.config
        target_model = copy.deepcopy(self.raw_model).to(self.device)
        target_model.train(False)

        def run_epoch(epoch_idx, loader):
            model.train(True)
            critic_model.train(train_critic)
            if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch_idx)

            metric_sums = {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "ratio": 0.0,
                "confidence": 0.0,
            }
            batch_count = 0

            for batch in loader:
                (s, o, a, r, ava, v, rtg, ret, adv, t, pre_a,
                 next_s, next_rtg, done) = batch

                s = s.to(self.device, non_blocking=True)
                o = o.to(self.device, non_blocking=True)
                a = a.to(self.device, non_blocking=True)
                r = r.to(self.device, non_blocking=True)
                ava = ava.to(self.device, non_blocking=True)
                v = v.to(self.device, non_blocking=True)
                rtg = rtg.to(self.device, non_blocking=True)
                ret = ret.to(self.device, non_blocking=True)
                adv = adv.to(self.device, non_blocking=True)
                t = t.to(self.device, non_blocking=True)
                pre_a = pre_a.to(self.device, non_blocking=True)
                next_s = next_s.to(self.device, non_blocking=True)
                next_rtg = next_rtg.to(self.device, non_blocking=True)
                done = done.to(self.device, non_blocking=True)

                del r, next_s, next_rtg, done

                with torch.amp.autocast('cuda', dtype=self._amp_dtype, enabled=self._use_amp):
                    logits, _ = model(o, pre_a, rtg, t)
                    if config.mode == "offline":
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a.reshape(-1))
                        entropy_info = 0.0
                        ratio_info = 0.0
                        confidence_info = 0.0
                    elif config.mode == "online":
                        adv = adv.reshape(-1, adv.size(-1))
                        # Normalize advantages
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                        logits = logits.masked_fill(ava == 0, -1e10)
                        distri = Categorical(logits=logits.reshape(-1, logits.size(-1)))
                        target_a = a.reshape(-1)
                        log_a = distri.log_prob(target_a).unsqueeze(-1)

                        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self._amp_dtype, enabled=self._use_amp):
                            old_logits, _ = target_model(o, pre_a, rtg, t)
                        old_logits = old_logits.masked_fill(ava == 0, -1e10)
                        old_distri = Categorical(logits=old_logits.reshape(-1, old_logits.size(-1)))
                        old_log_a = old_distri.log_prob(target_a).unsqueeze(-1)

                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * adv
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * adv
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)

                        act_entropy = distri.entropy().unsqueeze(-1)
                        loss = actor_loss - 0.08 * act_entropy

                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError

                    loss = loss.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                critic_loss_info = 0.0
                if train_critic:
                    with torch.amp.autocast('cuda', dtype=self._amp_dtype, enabled=self._use_amp):
                        v_value, _ = critic_model(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)
                        critic_loss = critic_loss.mean()
                        critic_loss_info = critic_loss.item()

                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.raw_critic_model.parameters(), config.grad_norm_clip)
                    self.critic_optimizer.step()

                metric_sums["actor_loss"] += loss.item()
                metric_sums["critic_loss"] += critic_loss_info
                metric_sums["entropy"] += entropy_info
                metric_sums["ratio"] += ratio_info
                metric_sums["confidence"] += confidence_info
                batch_count += 1

            return self._reduce_metrics(metric_sums, batch_count)

        loader = self._build_loader(dataset, 0)
        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0.0, 0.0, 0.0, 0.0, 0.0
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch(epoch, loader)
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence
