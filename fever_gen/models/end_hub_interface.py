# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, Iterator, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders, data_utils
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict


logger = logging.getLogger(__name__)


class EndHubInterface(GeneratorHubInterface):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/bart
    """

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        # self.cfg.dataset.max_tokens = 8192
        self.model = self.models[0]

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        # bpe_sentence = tokens
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample

    # def generate(
    #     self,
    #     tokenized_sentences: List[torch.LongTensor],
    #     *args,
    #     inference_step_args=None,
    #     skip_invalid_size_inputs=False,
    #     **kwargs
    # ) -> List[List[Dict[str, torch.Tensor]]]:
    #     inference_step_args = inference_step_args or {}
    #     if "prefix_tokens" in inference_step_args:
    #         raise NotImplementedError("prefix generation not implemented for BART")
    #     res = []
    #     for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
    #         src_tokens = batch['net_input']['src_tokens']
    #         inference_step_args["prefix_tokens"] =src_tokens.new_full(
    #             (src_tokens.size(0), 1), fill_value=self.task.source_dictionary.bos()
    #         ).to(device=self.device)
    #         results = super().generate(
    #             src_tokens,
    #             *args,
    #             inference_step_args=inference_step_args,
    #             skip_invalid_size_inputs=skip_invalid_size_inputs,
    #             **kwargs
    #         )
    #         res.extend(results)
    #     return res
    def generate(self, *args, retrieval_evidence=False, tokenized_titles=[], tokenized_documents=[], **kwargs) -> List[List[Dict[str, torch.Tensor]]]:
        if retrieval_evidence:
            return self.super_generate(*args, **kwargs, tokenized_titles=tokenized_titles, tokenized_documents=tokenized_documents)
        return super().generate(*args, **kwargs)

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(
        self,
        masked_inputs: List[str],
        topk: int = 5,
        match_source_len: bool = True,
        **generate_kwargs
    ):
        masked_token = '<mask>'
        batch_tokens = []
        for masked_input in masked_inputs:
            assert masked_token in masked_input, \
                "please add one {} token for the input".format(masked_token)

            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '<s> ' + text_spans_bpe + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            ).long()
            batch_tokens.append(tokens)

        # ensure beam size is at least as big as topk
        generate_kwargs['beam'] = max(
            topk,
            generate_kwargs.get('beam', -1),
        )
        generate_kwargs['match_source_len'] = match_source_len
        batch_hypos = self.generate(batch_tokens, **generate_kwargs)

        return [
            [(self.decode(hypo['tokens']), hypo['score']) for hypo in hypos[:topk]]
            for hypos in batch_hypos
        ]

    def sample(
        self,
        sentences: List[str],
        beam: int = 5,
        verbose: bool = False,
        text_to_id=None,
        marginalize=False,
        marginalize_lenpen=0.5,
        max_len_a=1024,
        max_len_b=1024,
        titles=None,
        documents=None,
        retrieval_evidence=False,
        **kwargs,
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        tokenized_documents = []
        tokenized_titles = []
        if documents is not None and titles is not None:
            tokenized_documents = []
            for docs in documents:
                docs = [self.encode(doc) for doc in docs]
                docs_tokens = self.merge_doc(
                    docs,
                    left_pad=False,
                    pad_to_length=None,
                )
                tokenized_documents.append(docs_tokens)
            tokenized_titles = [self.encode(title) for title in titles]
        batched_hypos = self.generate(
            tokenized_sentences,
            beam,
            verbose,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            tokenized_titles=tokenized_titles,
            tokenized_documents=tokenized_documents,
            retrieval_evidence=retrieval_evidence,
            **kwargs,
        )
        if retrieval_evidence:
            outputs = [
                [
                    {"text": hypo["tokens"].tolist(), "score": hypo["score"]}
                    for hypo in hypos
                ]
                for hypos in batched_hypos
            ]
        else:
            outputs = [
                [
                    {"text": self.decode(hypo["tokens"]), "score": hypo["score"]}
                    for hypo in hypos
                ]
                for hypos in batched_hypos
            ]
        if text_to_id:
            outputs = [
                [{**hypo, "id": text_to_id(hypo["text"])} for hypo in hypos]
                for hypos in outputs
            ]

            if marginalize:
                for (i, hypos), hypos_tok in zip(enumerate(outputs), batched_hypos):
                    outputs_dict = defaultdict(list)
                    for hypo, hypo_tok in zip(hypos, hypos_tok):
                        outputs_dict[hypo["id"]].append(
                            {**hypo, "len": len(hypo_tok["tokens"])}
                        )

                    outputs[i] = sorted(
                        [
                            {
                                "id": _id,
                                "texts": [hypo["text"] for hypo in hypos],
                                "scores": torch.stack(
                                    [hypo["score"] for hypo in hypos]
                                ),
                                "score": torch.stack(
                                    [
                                        hypo["score"]
                                        * hypo["len"]
                                        / (hypo["len"] ** marginalize_lenpen)
                                        for hypo in hypos
                                    ]
                                ).logsumexp(-1),
                            }
                            for _id, hypos in outputs_dict.items()
                        ],
                        key=lambda x: x["score"],
                        reverse=True,
                    )

        return outputs

    def merge_doc(self, docs, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [torch.LongTensor(v[:256]) for v in docs],
            1,
            None,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=1,
        )
    
    def _build_evidence_batches(
        self, tokens: List[List[int]], titles, docs, skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_evidence_inference(tokens, lengths, titles=titles, docs=docs),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def super_generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        tokenized_titles: List[torch.LongTensor] = None,
        tokenized_documents: List[torch.LongTensor] = None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                if k != "prefix_allowed_tokens_fn":
                    setattr(gen_args, k, v)
        generator = self.task.build_evidence_generator(
            self.models,
            gen_args,
            prefix_allowed_tokens_fn=kwargs.get("prefix_allowed_tokens_fn", None),
        )
        
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_evidence_batches(tokenized_sentences, tokenized_titles, tokenized_documents, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs
