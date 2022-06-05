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
        if label == 'NOT ENOUGH INFO':
            # continue
            source.append(claim)
            target.append(' [ NIL ] { NOT ENOUGH INFO }')
        else:
            e_dict = {}
            for i, evidence in enumerate(doc['evidence']):
                title, index, content = evidence
                if title not in e_dict:
                    e_dict[title] = [content]
                else:
                    e_dict[title].append(content)
            output = ''
            for title, contents in e_dict.items():
                contents = [f'{{ {c} }}' for c in contents]
                output = output + f' [ {title} ] ' + ' '.join(contents)
            # output = output + f' [ {title} ] {{ {index} }}'
            source.append(claim)
            target.append(output)

    return source, target, target


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
    split_name = os.path.basename(args.input_filename).split("_")[1].replace(".json", "")

    source, target, target_doc = convert_kilt_to_fairseq(
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
