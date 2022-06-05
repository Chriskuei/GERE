# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch

from .trie import DummyTrieContinueEvidence, DummyTrieMention, Trie


def get_prefix_allowed_tokens_fn(
    model,
    bpe = None,
    source_dictionary = None,
    max_positions = None,
    start_title_token="[",
    end_title_token="]",
    start_evidence_token="{",
    end_evidence_token="}",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    def encode(
            sentence: str, *addl_sentences, no_separator=True
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
        tokens = bpe.encode(sentence)
        if len(tokens.split(" ")) > min(max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(max_positions) - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + bpe.encode(s) + " </s>"
        tokens = source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            bpe.decode(source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: encode(x).tolist(),
        lambda x: decode(torch.tensor(x)),
        model.decoder.dictionary.bos(),
        model.decoder.dictionary.pad(),
        model.decoder.dictionary.eos(),
        len(model.decoder.dictionary),
        start_title_token,
        end_title_token,
        start_evidence_token,
        end_evidence_token,
        mention_trie,
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
    mention_trie: Trie = None,
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

    if mention_trie is None:
        mention_trie = DummyTrieMention(
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
        
        for tok in sent[::-1]:
            if tok == codes["end_evidence_token"]:
                # print('end e')
                return [codes["EOS"], codes["start_title_token"], codes["start_evidence_token"]]
            if tok == codes["start_evidence_token"]:
                # print('start e')
                # return [i for i in range(vocabulary_length)]
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
            ment_next = mention_trie.get(sent[pointer_start + 1 :])
        else:
            ment_next = mention_trie.get([])

        if codes["EOS"] in ment_next:
            ment_next.remove(codes["EOS"])
            ment_next.append(codes["end_title_token"])
        return ment_next

    def get_pointer_title(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_title_token"]:
                pointer_start = i
            elif e == codes["end_title_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_evidence(sent):
        return candidates_trie.get([])

    return prefix_allowed_tokens_fn
