# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, evidence_mask=None):
    # print('***ignore_index', ignore_index)
    # print('lprobs', lprobs)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    # print('before', target)
    if evidence_mask is not None:
        evidence_mask = evidence_mask.view(-1, 1)
        target = target.masked_fill(evidence_mask, ignore_index)
    # print('after', target)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # print('nll_loss', nll_loss)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # print('smooth_loss', smooth_loss)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        # print(pad_mask.shape, smooth_loss.shape, nll_loss.shape)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    # print(nll_loss, smooth_loss)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    valid_mask = ~pad_mask
    sample_size = valid_mask.sum()
    return loss, nll_loss, sample_size


@register_criterion(
    "fever_connection_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedConnectionCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # print('net_output', net_output)
        loss, nll_loss, sample_size, e_loss, e_nll_loss, e_sample_size = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        # print('sample_size', sample_size, e_sample_size)
        # print(loss, nll_loss, e_loss, e_nll_loss)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "sample_size": sample_size,
            "e_loss": e_loss.data,
            "e_nll_loss": e_nll_loss.data,
            "e_sample_size": e_sample_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
        }
        alpha, beta = net_output[-2:]
        # print(alpha, beta)
        # loss = alpha * (loss / sample_size) + beta * (e_loss / e_sample_size)
        # nll_loss = beta * (nll_loss / sample_size) + beta * (e_nll_loss / e_sample_size)
        loss = (loss / sample_size) + (e_loss / e_sample_size)
        nll_loss = (nll_loss / sample_size) + (e_nll_loss / e_sample_size)
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        evidence = net_output[2]
        evidence_mask = net_output[3]
        evidence_target = net_output[4]
        ignore_index = evidence.shape[-1] - 1

        loss, nll_loss, sample_size = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            # evidence_mask=evidence_mask
        )

        e_loss, e_nll_loss, e_sample_size = label_smoothed_nll_loss(
            evidence.view(-1, evidence.size(-1)),
            evidence_target.view(-1, 1),
            self.eps,
            ignore_index=ignore_index,
            reduce=reduce,
            # evidence_mask=~evidence_mask
        )
        # print('e_loss, e_nll_loss', e_loss, e_nll_loss)
        return loss, nll_loss, sample_size, e_loss, e_nll_loss, e_sample_size

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
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

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
