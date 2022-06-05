# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import jsonlines
from tqdm import tqdm


def convert_kilt_to_fairseq(dataset):

    source = []
    target = []
    for doc in tqdm(dataset, desc="Processing"):
        claim = doc['claim']
        label = doc['label']
        if label == 'NOT ENOUGH INFO' and doc['evidence'] != []:
            source.append(claim)
            # print(claim)
            evidence = doc['evidence'][0]
            title, index, sentence, lines = evidence[0], evidence[1], evidence[2], evidence[-1]
            evidence_texts = [line.split("\t")[:2] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            map_num = {int(item[0]): idx for idx, item in enumerate(evidence_texts)}
            # output = f' [ {title} ] {{ {index} }} < 0 >'
            idx = map_num[index]
            output = f' [ {title} ] {{ {idx} }}'
            target.append(output)
            # target.append(' [ NIL ] { NOT ENOUGH INFO }')
        else:
            e_dict = {}
            for i, evidence in enumerate(doc['evidence']):
                title, index, sentence, lines = evidence[0], evidence[1], evidence[2], evidence[-1]
                evidence_texts = [line.split("\t")[:2] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
                # print(evidence_texts)
                map_num = {int(item[0]): str(idx) for idx, item in enumerate(evidence_texts) if item[0]!=''}
                if title not in e_dict:
                    e_dict[title] = [map_num[index]]
                else:
                    e_dict[title].append(map_num[index])
            output = ''
            for title, contents in e_dict.items():
                contents = ' '.join(contents)
                # contents = [f'{{ {c} }}' for c in contents]
                output = output + f' [ {title} ] {{ {contents} }}'
            # output = output + f' [ {title} ] {{ {index} }}'
            # output = output + ' < 1 >'
            output = output
            source.append(claim)
            target.append(output)

    return source, target


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_filename",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the converted dataset",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    logging.info("Loading {}".format(args.input_filename))
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]
    split_name = os.path.basename(args.input_filename).split(".")[0].replace(".json", "")

    source, target = convert_kilt_to_fairseq(
        dataset,
    )

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for type_name, data in (("source", source), ("target", target)):

        with open(
            os.path.join(
                args.output_path,
                "{}.{}".format(split_name, type_name),
            ),
            "w",
        ) as f:
            f.writelines(
                [doc.replace("\r", ">>").replace("\n", ">>") + "\n" for doc in data]
            )
