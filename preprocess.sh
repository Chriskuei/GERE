#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


DATASET_PATH=$1
MODEL_PATH=$2

echo "Processing ${DATASET}"

# cd ../fairseq

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "target"; do
        python preprocess/multiprocessing_bpe_encoder_pipeline.py \
            --encoder-json "$MODEL_PATH/encoder.json" \
            --vocab-bpe "$MODEL_PATH//vocab.bpe" \
            --inputs "$DATASET_PATH/$SPLIT.$LANG" \
            --outputs "$DATASET_PATH/$SPLIT.bpe.$LANG" \
            --doc_inputs "$DATASET_PATH/$SPLIT.doc.json" \
            --doc_outputs "$DATASET_PATH/$SPLIT.bpe.doc.json" \
            --workers 60 \
            --keep-empty;
    done
done

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source"; do
        python -m preprocess.multiprocessing_bpe_encoder\
            --encoder-json "$MODEL_PATH/encoder.json" \
            --vocab-bpe "$MODEL_PATH//vocab.bpe" \
            --inputs "$DATASET_PATH/$SPLIT.$LANG" \
            --outputs "$DATASET_PATH/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --user-dir fever_gen \
    --task fever \
    --trainpref "$DATASET_PATH/train.bpe" \
    --validpref "$DATASET_PATH/dev.bpe" \
    --destdir "$DATASET_PATH/bin" \
    --workers 60 \
    --srcdict "$MODEL_PATH//dict.source.txt" \
    --tgtdict "$MODEL_PATH/dict.target.txt" \
    --process-doc \
    --train-doc "$DATASET_PATH/train.bpe.doc.json" \
    --dev-doc "$DATASET_PATH/dev.bpe.doc.json" \
