#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
DATASET=$1
NAME=$2

fairseq-generate $DATASET/bin/ \
    --task ir \
    --user-dir fever_gen \
    --max-tokens 4096 \
    --path model/$NAME/checkpoint_best.pt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 200 \
    --remove-bpe \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test  \
    --results-path predictions/$NAME

# grep ^H result.txt | sort -n -k 2 -t '-' | cut -f 3