# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from requests.api import get

import torch

from fever_gen.trie import DummyTrieContinueEvidence, DummyTrieMention, Trie


def get_end_to_end_prefix_allowed_tokens_fn_hf(
    model,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    split_token=".",
    title_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x)),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        len(model.tokenizer) - 1,
        start_title_token,
        end_title_token,
        start_evidence_token,
        end_evidence_token,
        split_token,
        title_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def get_prefix_allowed_tokens_fn(
    model,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    split_token=".",
    title_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        start_title_token,
        end_title_token,
        start_evidence_token,
        end_evidence_token,
        split_token,
        title_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    split_token=".",
    title_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "start_title_token",
                "end_title_token",
                "start_evidence_token",
                "end_evidence_token",
                "split_token",
                "BOS",
            ),
            (
                start_title_token,
                end_title_token,
                start_evidence_token,
                end_evidence_token,
                split_token,
                bos_token_id,
            ),
        )
    }
    codes["EOS"] = eos_token_id
    print(codes)

    if title_trie is None:
        title_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )

    if candidates_trie is None and mention_to_candidates_dict is None:
        candidates_trie = DummyTrieContinueEvidence(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                    codes['start_title_token'],
                    codes['end_title_token'],
                    codes['start_evidence_token'],
                )
            ],
            codes,
        )


    def prefix_allowed_tokens_fn(batch_id, sent):

        sent = sent.tolist()
        # print(status, sent)
        # print(sent)
        
        # for tok in sent[::-1]:
        #     if tok == codes["split_token"]:
        #         # print('start t')
        #         return get_trie_title(sent)
        available_tokens = get_trie_title(sent)
        # print('**', available_tokens)
        return available_tokens

    def get_trie_title(sent):

        pointer_start, _ = get_pointer_title(sent)
        if pointer_start + 1 < len(sent):
            title_next = title_trie.get(sent[pointer_start + 1 :])
        else:
            title_next = title_trie.get([])

        if codes["EOS"] in title_next:
            # title_next.remove(codes["EOS"])
            title_next.append(codes["split_token"])
        return title_next

    def get_pointer_title(sent):
        pointer_end = -1
        pointer_start = 0
        for i, e in enumerate(sent):
            if e == codes["split_token"]:
                pointer_start = i

        return pointer_start, pointer_end

    def get_trie_evidence(sent):
        return candidates_trie.get([])

    return prefix_allowed_tokens_fn
