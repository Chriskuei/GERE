
#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
DATASET=$1
NAME=$2

fairseq-train $DATASET/bin/ \
    --fp16 \
    --fp16-scale-tolerance 0.25 \
    --user-dir fever_gen \
    --save-dir models/$NAME \
    --no-epoch-checkpoints \
    --tensorboard-logdir tensorboard_logs/$NAME \
    --arch bart_large_end \
    --task end  \
    --criterion fever_connection_label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 4096 \
    --update-freq 8  \
    --max-update 20000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 20000  \
    --warmup-updates 2000  \
    --ddp-backend no_c10d  \
    --num-workers 56  \
    --reset-meters  \
    --reset-optimizer \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 200 \
    --validate-interval-updates 4000 \
    --reset-dataloader > models/log/$NAME.log \
    --restore-file models/bart.large/model.pt \
    # --restore-file ~/research/GENRE-main/models/fairseq_wikipage_retrieval/model.pt \


# CUDA_VISIBLE_DEVICES=1 fairseq-train data-bin/iwslt14.tokenized.de-en \
#     --fp16 \
#     --arch tutorial_simple_lstm \
#     --encoder-dropout 0.2 --decoder-dropout 0.2 \
#     --optimizer adam --lr 0.005 --lr-shrink 0.5 \
#     --max-tokens 12000 \
#     --user-dir fever_gen \
