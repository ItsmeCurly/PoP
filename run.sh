#!/usr/bin/env bash

set -xe

SQUAD_DIR=data/squad11
POLICYQA_DIR=data/policyqa
PRIVACYQA_DIR=data/privacyqa

base_nosquad=bert-base-uncased
base_squad=csarron/bert-base-uncased-squad-v1

policyqa_nosquad_out=models/bert-base-uncased-policyqa
policyqa_squad_out=models/bert-base-uncased-policyqa-squad
privacyqa_nosquad_out=models/bert-base-uncased-privacyqa
privacyqa_squad_out=models/bert-base-uncased-privacyqa-squad

epochs=50

max_seq_length=384
doc_stride=128
train_batch_size=8
eval_batch_size=16
save_steps=500


function train_policyqa_nosquad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_nosquad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $POLICYQA_DIR/train.json \
    --predict_file $POLICYQA_DIR/dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$policyqa_nosquad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size  \
    --per_gpu_train_batch_size=$train_batch_size   \
    --evaluate_during_training \
    --save_steps $save_steps
}

function evaluate_policyqa_nosquad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $policyqa_nosquad_out \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$policyqa_nosquad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size \
    --eval_all_checkpoints
}

function train_policyqa_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $POLICYQA_DIR/train.json \
    --predict_file $POLICYQA_DIR/dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$policyqa_squad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size  \
    --per_gpu_train_batch_size=$train_batch_size   \
    --evaluate_during_training \
    --save_steps $save_steps
}

function evaluate_policyqa_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $policyqa_squad_out \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$policyqa_squad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size \
    --eval_all_checkpoints
}

function train_privacyqa_nosquad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_nosquad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $PRIVACYQA_DIR/policy_train_squad.json \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --learning_rate 3e-5 \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$privacyqa_nosquad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size  \
    --per_gpu_train_batch_size=$train_batch_size   \
    --evaluate_during_training \
    --save_steps $save_steps
}

function evaluate_privacyqa_nosquad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $privacyqa_nosquad_out \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$privacyqa_nosquad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size \
    --eval_all_checkpoints
}

function train_privacyqa_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $PRIVACYQA_DIR/policy_train_squad.json \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --learning_rate 3e-5 \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$privacyqa_squad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size  \
    --per_gpu_train_batch_size=$train_batch_size   \
    --evaluate_during_training \
    --save_steps $save_steps
}

function evaluate_privacyqa_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $privacyqa_squad_out \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --output_dir ./$privacyqa_squad_out/ \
    --per_gpu_eval_batch_size=$eval_batch_size \
    --eval_all_checkpoints
}

rm -r models/
rm -r runs/

rm cached_*

train_policyqa_nosquad
evaluate_policyqa_nosquad

train_policyqa_squad
evaluate_policyqa_squad

train_privacyqa_nosquad
evaluate_privacyqa_nosquad

train_privacyqa_squad
evaluate_privacyqa_squad
