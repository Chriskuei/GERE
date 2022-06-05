# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, List

import torch

from fever_gen.trie import DummyTrieContinueEvidence, DummyTrieMention, Trie
from fever_gen.retrieval import FeverDocDB


def get_end_to_end_prefix_allowed_tokens_fn_hf(
    model,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    title_trie: Trie = None,
    db_path: str = None,
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
        title_trie,
        db_path,
        candidates_trie,
        mention_to_candidates_dict,
    )


def get_prefix_allowed_tokens_fn(
    model,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    title_trie: Trie = None,
    db_path: str = None,
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
        title_trie,
        db_path,
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
    title_trie: Trie = None,
    db_path: str = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    db = FeverDocDB(db_path)

    def get_doc_from_db(title):
        doc_lines = db.get_doc_lines(title)
        return doc_lines
    
    def get_doc_trie_from_db(title):
        doc_trie = db.get_doc_trie(title)
        doc_trie = json.loads(doc_trie, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        return doc_trie

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
            ),
            (
                start_title_token,
                end_title_token,
                start_evidence_token,
                end_evidence_token,
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
        # print('****', sent)
        # print(decode_fn(sent))
        
        for tok in sent[::-1]:
            if tok == codes["end_evidence_token"]:
                # print('end e')
                return [codes["EOS"], codes["start_title_token"], codes["start_evidence_token"]]
            if tok == codes["start_evidence_token"]:
                # print('start e')
                return get_trie_evidence(sent)
            if tok == codes["end_title_token"]:
                # print('end t')
                return codes["start_evidence_token"]
            if tok == codes["start_title_token"]:
                # print('start t')
                return get_trie_title(sent)
        return codes["start_title_token"]

    def get_trie_title(sent):

        pointer_start, _ = get_pointer_title(sent)
        if pointer_start + 1 < len(sent):
            ment_next = title_trie.get(sent[pointer_start + 1 :])
        else:
            ment_next = title_trie.get([])

        if codes["EOS"] in ment_next:
            ment_next.remove(codes["EOS"])
            ment_next.append(codes["end_title_token"])
        return ment_next

    def get_pointer_evidence(sent):
        len_sent = len(sent) - 1
        pointer_end = -1
        for i, e in enumerate(sent[::-1]):
            if e == codes["start_evidence_token"]:
                pointer_start = len_sent - i
                break
            elif e == codes["end_evidence_token"]:
                pointer_end = len_sent - i

        return pointer_start, pointer_end

    def get_pointer_title(sent):
        len_sent = len(sent) - 1
        pointer_end = -1
        for i, e in enumerate(sent[::-1]):
            if e == codes["start_title_token"]:
                pointer_start = len_sent - i
                break
            elif e == codes["end_title_token"]:
                pointer_end = len_sent - i

        return pointer_start, pointer_end

    def get_trie_evidence(sent):

        title_start, title_end = get_pointer_title(sent)
        title = decode_fn(sent[title_start + 1 : title_end]).strip()

        evidence_trie = get_doc_trie_from_db(title)
        evidence_trie = Trie.load_from_dict(evidence_trie)

        evidence_start, _ = get_pointer_evidence(sent)
        if evidence_start + 1 < len(sent):
            evidence_next = evidence_trie.get(sent[evidence_start + 1 :])
        else:
            evidence_next = evidence_trie.get([])

        if codes["EOS"] in evidence_next:
            evidence_next.remove(codes["EOS"])
            evidence_next.append(codes["end_evidence_token"])
        return [int(i) for i in evidence_next]
        return evidence_next

    return prefix_allowed_tokens_fn
