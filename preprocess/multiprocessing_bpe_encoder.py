#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
import json
from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--doc_inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--doc_outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]
        print(args.doc_inputs)
        if args.doc_inputs != ["-"]:
            doc_inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-"
                else sys.stdin
                for input in args.doc_inputs
            ]
            doc_outputs = [
                stack.enter_context(open(output, "w", encoding="utf-8"))
                if output != "-"
                else sys.stdout
                for output in args.doc_outputs
            ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        if args.doc_inputs != ["-"]:
            encoded_lines = pool.imap(encoder.encode_docs, zip(*inputs, *doc_inputs), 100)
        else:
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines, enc_docs) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
                if args.doc_inputs != ["-"]:
                    for enc_docs, output_h in zip(enc_docs, doc_outputs):
                        print(json.dumps(enc_docs), file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args
        self.start_title_token = "["
        self.end_title_token = "]"
        self.start_evidence_token = "{"
        self.end_evidence_token = "}"

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_tokens(self):
        self.start_title_id = self.encode(' ' + self.start_title_token)[0]
        self.end_title_id = self.encode(' ' + self.end_title_token)[0]
        self.start_evidence_id = self.encode(' ' + self.start_evidence_token)[0]
        self.end_evidence_id = self.encode(' ' + self.end_evidence_token)[0]

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip('\n')
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None, None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines, None]

    def encode_docs(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        self.encode_tokens()
        lines, doc_lines = lines
        enc_lines = []
        enc_docs = []

        if not isinstance(lines, list):
            lines = [lines]
        if not isinstance(doc_lines, list):
            doc_lines = [doc_lines]

        for line, doc_lines in zip(lines, doc_lines):
            line = line.rstrip("\n")
            doc = json.loads(doc_lines)
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None, None]
            tokens = self.encode(line)

            evidence_dict = {}
            start_evidence = False
            title_num = 0
            for index, token in enumerate(tokens):
                if token == self.start_title_id:
                    title_num += 1
                    evidence_dict[title_num] = {}
                    evidence_dict[title_num]["position"] = []
                    evidence_dict[title_num]['evidence_id'] = []
                elif token == self.start_evidence_id:
                    start_evidence = True
                elif token == self.end_evidence_id:
                    start_evidence = False
                elif start_evidence:
                    evidence_dict[title_num]['position'].append(index)
            evidences = doc['evidence']
            if doc['label'] == 'NOT ENOUGH INFO' and evidences != []:
                evidences = [evidences[0]]
            title_to_id = {}
            title_id = 0
            doc_text = {}
            for evidence in evidences:
                title, index, sentence, lines = evidence[0], evidence[1], evidence[2], evidence[-1]
                evidence_texts = [line.split("\t")[:2] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
                if title not in title_to_id:
                    title_id += 1
                    title_to_id[title] = title_id
                doc_text = {idx: self.encode(f' {idx} ' + text) for idx, text in evidence_texts}
                evidence_dict[title_to_id[title]]['doc'] = doc_text
                evidence_dict[title_to_id[title]]['evidence_id'].append(index)

            enc_lines.append(" ".join(tokens))
            enc_docs.append(evidence_dict)
        return ["PASS", enc_lines, enc_docs]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
