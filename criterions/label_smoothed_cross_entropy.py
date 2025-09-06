# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class AdjustLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    det_weight: float = field(
        default=1.0,
        metadata={"help": "weight of detection loss"},
    )
    cls_weight: float = field(
        default=1.0,
        metadata={"help": "weight of classification loss"},
    )

    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Ignore eos token"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    drop_worst_ratio: float = field(
        default=0.0,
        metadata={"help": "ratio for discarding bad samples"},
    )
    drop_worst_after: int = field(
        default=0,
        metadata={"help": "steps for discarding bad samples"},
    )
    use_rdrop: bool = field(
        default=False, metadata={"help": "use R-Drop"}
    )
    reg_alpha: float = field(
        default=1.0, metadata={"help": "weight for R-Drop"}
    )
    sample_patch_num: int = field(
        default=196, metadata={"help": "sample patches for v1"}
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    # Auxiliary GMM loss controls
    gmm_aux_init: float = field(
        default=0.0,
        metadata={"help": "initial coefficient for auxiliary GMM loss (0 to disable)"}
    )
    gmm_aux_start: int = field(
        default=0,
        metadata={"help": "update step to start annealing auxiliary GMM loss"}
    )
    gmm_aux_end: int = field(
        default=50000,
        metadata={"help": "update step to finish annealing auxiliary GMM loss (0 -> immediate off)"}
    )
    gmm_weight_coef: float = field(
        default=1e-3,
        metadata={"help": "coefficient for weight-uniformity term (KL to uniform)"}
    )
    gmm_sep_coef: float = field(
        default=1e-3,
        metadata={"help": "coefficient for mean-separation term"}
    )
    gmm_sep_margin: float = field(
        default=1.0,
        metadata={"help": "hinge margin for normalized mean separation"}
    )


def construct_rdrop_sample(x):
    if isinstance(x, dict):
        for key in x:
            x[key] = construct_rdrop_sample(x[key])
        return x
    elif isinstance(x, torch.Tensor):
        return x.repeat(2, *([1] * (x.dim() - 1)))
    elif isinstance(x, int):
        return x * 2
    elif isinstance(x, np.ndarray):
        return x.repeat(2)
    else:
        raise NotImplementedError


def kl_loss(p, q):
    p_loss = F.kl_div(p, torch.exp(q), reduction='sum')
    q_loss = F.kl_div(q, torch.exp(p), reduction='sum')
    loss = (p_loss + q_loss) / 2
    return loss


def _linear_anneal_coeff(update_num: int, start_step: int, end_step: int, init_coeff: float) -> float:
    if init_coeff <= 0.0:
        return 0.0
    if end_step <= start_step:
        return 0.0
    if update_num <= start_step:
        return float(init_coeff)
    if update_num >= end_step:
        return 0.0
    ratio = (update_num - start_step) / float(end_step - start_step)
    return float(init_coeff) * (1.0 - ratio)


@torch.no_grad()
def _safe_mean(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return torch.tensor(0.0, device=t.device, dtype=t.dtype)
    return t.mean()


def compute_gmm_aux_loss(
        w: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        sep_margin: float = 1.0,
        weight_coef: float = 1e-3,
        sep_coef: float = 1e-3,
        eps: float = 1e-8,
):
    """Compute auxiliary GMM loss encouraging uniform weights and separated means.

    Args:
        w: shape [N, K]
        mu: shape [N, K, D]
        sigma: shape [N, K, D]
    Returns:
        aux_loss: scalar tensor
        weight_kl: scalar tensor (KL(w || Uniform))
        sep_loss: scalar tensor
    """
    if w.numel() == 0:
        device = mu.device if mu.is_cuda or mu.device.type != 'cpu' else w.device
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero

    # Ensure float32 for numeric stability
    w_f = w.float()
    mu_f = mu.float()
    sigma_f = sigma.float()

    # KL to uniform for weights
    K = w_f.size(-1)
    logK = math.log(K)
    weight_kl_per = (w_f * (torch.log(w_f.clamp_min(eps)) + logK)).sum(dim=-1)
    weight_kl = weight_kl_per.mean()

    # Hinge repulsion on means with variance normalization
    # pairwise normalized squared distances between component means
    # Shapes: [N, K, 1, D] - [N, 1, K, D] -> [N, K, K, D]
    diff = mu_f[:, :, None, :] - mu_f[:, None, :, :]
    var_sum = sigma_f[:, :, None, :].pow(2) + sigma_f[:, None, :, :].pow(2) + eps
    d2 = (diff.pow(2) / var_sum).sum(dim=-1)  # [N, K, K]

    # Use only i<j pairs
    K_idx = d2.size(-1)
    if K_idx >= 2:
        tri_mask = torch.triu(torch.ones((K_idx, K_idx), dtype=torch.bool, device=d2.device), diagonal=1)
        d2_pairs = d2[:, tri_mask]
        d_pairs = torch.sqrt(d2_pairs + eps)
        sep_per = F.relu(sep_margin - d_pairs).pow(2)
        sep_loss = sep_per.mean()
    else:
        sep_loss = torch.tensor(0.0, device=d2.device)

    aux_loss = weight_coef * weight_kl + sep_coef * sep_loss
    return aux_loss, weight_kl, sep_loss


def label_smoothed_nll_loss(
        lprobs, target, epsilon, update_num, reduce=True,
        drop_worst_ratio=0.0, drop_worst_after=0, use_rdrop=False, reg_alpha=1.0,
        constraint_masks=None, constraint_start=None, constraint_end=None
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    if constraint_masks is not None:
        smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
    elif constraint_start is not None and constraint_end is not None:
        constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
        smooth_loss = -lprobs[:, constraint_range].sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (len(constraint_range) - 1 + 1e-6)
    else:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if drop_worst_ratio > 0 and update_num > drop_worst_after:
        if use_rdrop:
            true_batch_size = loss.size(0) // 2
            _, indices = torch.topk(loss[:true_batch_size], k=int(true_batch_size * (1 - drop_worst_ratio)), largest=False)
            loss = torch.cat([loss[indices], loss[indices+true_batch_size]])
            nll_loss = torch.cat([nll_loss[indices], nll_loss[indices+true_batch_size]])
            lprobs = torch.cat([lprobs[indices], lprobs[indices+true_batch_size]])
        else:
            loss, indices = torch.topk(loss, k=int(loss.shape[0] * (1 - drop_worst_ratio)), largest=False)
            nll_loss = nll_loss[indices]
            lprobs = lprobs[indices]


    ntokens = loss.numel()
    nll_loss = nll_loss.sum()

    loss = loss.sum()
    if use_rdrop:
        true_batch_size = lprobs.size(0) // 2
        p = lprobs[:true_batch_size]
        q = lprobs[true_batch_size:]
        if constraint_start is not None and constraint_end is not None:
            constraint_range = [0, 1, 2, 3] + list(range(constraint_start, constraint_end))
            p = p[:, constraint_range]
            q = q[:, constraint_range]
        loss += kl_loss(p, q) * reg_alpha

    return loss, nll_loss, ntokens

@register_criterion(
    "adjust_label_smoothed_cross_entropy", dataclass=AdjustLabelSmoothedCrossEntropyCriterionConfig
)
class AdjustLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            ignore_eos=False,
            report_accuracy=False,
            drop_worst_ratio=0,
            drop_worst_after=0,
            use_rdrop=False,
            reg_alpha=1.0,
            sample_patch_num=196,
            constraint_range=None,
            det_weight=1.0,
            cls_weight=1.0,
            gmm_aux_init=0.0,
            gmm_aux_start=0,
            gmm_aux_end=50000,
            gmm_weight_coef=1e-3,
            gmm_sep_coef=1e-3,
            gmm_sep_margin=1.0
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.ignore_eos = ignore_eos
        self.report_accuracy = report_accuracy
        self.drop_worst_ratio = drop_worst_ratio
        self.drop_worst_after = drop_worst_after
        self.use_rdrop = use_rdrop
        self.reg_alpha = reg_alpha
        self.sample_patch_num = sample_patch_num

        self.det_weight = det_weight
        self.cls_weight = cls_weight

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        # Aux GMM loss settings
        self.gmm_aux_init = gmm_aux_init
        self.gmm_aux_start = gmm_aux_start
        self.gmm_aux_end = gmm_aux_end
        self.gmm_weight_coef = gmm_weight_coef
        self.gmm_sep_coef = gmm_sep_coef
        self.gmm_sep_margin = gmm_sep_margin

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if isinstance(sample, list):
            if self.sample_patch_num > 0:
                sample[0]['net_input']['sample_patch_num'] = self.sample_patch_num
            loss_v1, sample_size_v1, logging_output_v1 = self.forward(model, sample[0], update_num, reduce)
            loss_v2, sample_size_v2, logging_output_v2 = self.forward(model, sample[1], update_num, reduce)
            loss = loss_v1 / sample_size_v1 + loss_v2 / sample_size_v2
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "loss_v1": loss_v1.data,
                "loss_v2": loss_v2.data,
                "nll_loss": logging_output_v1["nll_loss"].data / sample_size_v1 + logging_output_v2[
                    "nll_loss"].data / sample_size_v2,
                "ntokens": logging_output_v1["ntokens"] + logging_output_v2["ntokens"],
                "nsentences": logging_output_v1["nsentences"] + logging_output_v2["nsentences"],
                "sample_size": 1,
                "sample_size_v1": sample_size_v1,
                "sample_size_v2": sample_size_v2,
                "gt_prob": (logging_output_v1.get("gt_prob", 0) + logging_output_v2.get("gt_prob", 0)) / 2,
            }
            return loss, sample_size, logging_output

        if self.use_rdrop:
            construct_rdrop_sample(sample)

        net_output = model(**sample["net_input"])
        loss, nll_loss, ntokens, gt_prob, gmm_aux, gmm_w_kl, gmm_sep, gmm_aux_coeff = self.compute_loss(
            model, net_output, sample, update_num, det_weight=self.det_weight,
            cls_weight=self.cls_weight, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0)
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "gt_prob": gt_prob.data,
            "gmm_aux": gmm_aux.data if isinstance(gmm_aux, torch.Tensor) else torch.tensor(gmm_aux),
            "gmm_w_kl": gmm_w_kl.data if isinstance(gmm_w_kl, torch.Tensor) else torch.tensor(gmm_w_kl),
            "gmm_sep": gmm_sep.data if isinstance(gmm_sep, torch.Tensor) else torch.tensor(gmm_sep),
            "gmm_aux_coeff": torch.tensor(gmm_aux_coeff),
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        conf = sample['conf'][:, None, None] if 'conf' in sample and sample['conf'] is not None else 1
        constraint_masks = None
        if "constraint_masks" in sample and sample["constraint_masks"] is not None:
            constraint_masks = sample["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True) * conf
        target = sample["token_type"]
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, self.ignore_prefix_size:, :].contiguous()
        if self.ignore_eos:
            bsz, seq_len, embed_dim = lprobs.size()
            eos_indices = target.eq(self.task.tgt_dict.eos())
            lprobs = lprobs[~eos_indices].reshape(bsz, seq_len - 1, embed_dim)
            target = target[~eos_indices].reshape(bsz, seq_len - 1)
            if constraint_masks is not None:
                constraint_masks = constraint_masks[~eos_indices].reshape(bsz, seq_len - 1, embed_dim)
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))

        # index = torch.zeros(lprobs.shape[:2]).to(lprobs.device)
        # index[:, :4] = 1   # 1 indicates the location of detection results

        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), constraint_masks, None  # index.view(-1)

    def compute_loss(self, model, net_output, sample, update_num, det_weight=1.0, cls_weight=1.0, reduce=True):
        b = sample['target'].shape[0]
        lprobs, target, constraint_masks, index = self.get_lprobs_and_target(model, net_output, sample)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[target != -1]
        # index = index[target != self.padding_idx]
        lprobs = lprobs[target != -1]
        target = target[target != -1]

        loss_cls, nll_loss, ntokens = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            update_num,
            reduce=reduce,
            drop_worst_ratio=self.drop_worst_ratio,
            drop_worst_after=self.drop_worst_after,
            use_rdrop=self.use_rdrop,
            reg_alpha=self.reg_alpha,
            constraint_masks=constraint_masks,
            constraint_start=self.constraint_start,
            constraint_end=self.constraint_end
        )
        loss_cls = cls_weight * loss_cls/b

        # compute regression loss via GMM likelihood
        token_type = sample["token_type"]
        target_full = sample["target"]
        index = torch.zeros_like(target_full[:, :, 0]).to(target_full.device)
        index[:, :2] = 1  # first two tokens are detection
        mask = token_type == 0
        target = target_full[mask]
        det_mask = index[mask] > 0
        w, mu, sigma = net_output[1]
        w = w[mask]
        mu = mu[mask]
        sigma = sigma[mask]

        # Compute GMM likelihood in float32 for numerical stability, and sum logs instead of log of product
        with torch.cuda.amp.autocast(enabled=False):
            target_f = target.float()
            mu_f = mu.float()
            sigma_f = sigma.float()
            w_f = w.float()
            diff = (target_f.unsqueeze(1) - mu_f) / sigma_f
            quad = 0.5 * diff.pow(2).sum(-1)
            log_sigma_sum = torch.log(sigma_f).sum(-1)
            log_gauss = -quad - log_sigma_sum - math.log(2 * math.pi)
            log_prob = torch.log(w_f + 1e-9) + log_gauss
            nll = -torch.logsumexp(log_prob, dim=-1)
        prob = torch.exp(-nll)
        loss_reg = det_weight * nll[det_mask].mean() if det_mask.any() else 0.0
        if (~det_mask).any():
            loss_reg += nll[~det_mask].mean()

        # Auxiliary GMM loss with annealing
        aux_coeff = _linear_anneal_coeff(update_num, self.gmm_aux_start, self.gmm_aux_end, self.gmm_aux_init)
        if aux_coeff > 0.0 and w.numel() > 0:
            with torch.cuda.amp.autocast(enabled=False):
                gmm_aux_loss, gmm_w_kl, gmm_sep = compute_gmm_aux_loss(
                    w, mu, sigma,
                    sep_margin=self.gmm_sep_margin,
                    weight_coef=self.gmm_weight_coef,
                    sep_coef=self.gmm_sep_coef,
                )
            loss = loss_reg + loss_cls + aux_coeff * gmm_aux_loss
        else:
            gmm_aux_loss = torch.tensor(0.0, device=lprobs.device)
            gmm_w_kl = torch.tensor(0.0, device=lprobs.device)
            gmm_sep = torch.tensor(0.0, device=lprobs.device)
            loss = loss_reg + loss_cls
        if update_num % 5000 == 1:
            print(f"loss_reg: {loss_reg.item()} loss_cls: {loss_cls.item()} gmm_aux: {gmm_aux_loss.item()} coeff: {aux_coeff}")

        return loss, nll_loss, ntokens, prob.mean(), gmm_aux_loss, gmm_w_kl, gmm_sep, aux_coeff

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_v1 = sum(log.get("loss_v1", 0) for log in logging_outputs)
        loss_sum_v2 = sum(log.get("loss_v2", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_v1 = sum(log.get("sample_size_v1", 0) for log in logging_outputs)
        sample_size_v2 = sum(log.get("sample_size_v2", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "loss_v1", loss_sum_v1 / max(sample_size_v1, 1), max(sample_size_v1, 1), round=3
        )
        metrics.log_scalar(
            "loss_v2", loss_sum_v2 / max(sample_size_v2, 1), max(sample_size_v2, 1), round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v1", sample_size_v1, 1, round=3
        )
        metrics.log_scalar(
            "sample_size_v2", sample_size_v2, 1, round=3
        )

        gt_prob_sum = sum(log.get("gt_prob", 0) for log in logging_outputs)
        metrics.log_scalar("gt_prob", gt_prob_sum / sample_size, sample_size, round=6)

        # Auxiliary GMM metrics
        gmm_aux_sum = sum(log.get("gmm_aux", 0) for log in logging_outputs)
        gmm_w_kl_sum = sum(log.get("gmm_w_kl", 0) for log in logging_outputs)
        gmm_sep_sum = sum(log.get("gmm_sep", 0) for log in logging_outputs)
        gmm_aux_coeff_sum = sum(log.get("gmm_aux_coeff", 0) for log in logging_outputs)
        metrics.log_scalar("gmm_aux", gmm_aux_sum / max(sample_size, 1), max(sample_size, 1), round=6)
        metrics.log_scalar("gmm_w_kl", gmm_w_kl_sum / max(sample_size, 1), max(sample_size, 1), round=6)
        metrics.log_scalar("gmm_sep", gmm_sep_sum / max(sample_size, 1), max(sample_size, 1), round=6)
        metrics.log_scalar("gmm_aux_coeff", gmm_aux_coeff_sum / max(sample_size, 1), max(sample_size, 1), round=6)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
