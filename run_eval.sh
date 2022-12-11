#!/usr/bin/env bash

set -xe

SQUAD_DIR=data/squad11
POLICYQA_DIR=data/policyqa
PRIVACYQA_DIR=data/privacyqa

# Base models

base_nosquad=bert-base-uncased
base_squad=csarron/bert-base-uncased-squad-v1

# Output directories

pol=models/bert-base-uncased-policyqa
polsquad=models/bert-base-uncased-policyqa-squad

pri=models/bert-base-uncased-privacyqa
prisquad=models/bert-base-uncased-privacyqa-squad

squad=models/bert-base-uncased-squad

max_seq_length=384
doc_stride=128
train_batch_size=8
eval_batch_size=16
save_steps=500

# PolicyQA Evals

function evaluate_pol_policyqad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pol \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/dev.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pol/policyqad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_pol_policyqat () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pol \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pol/policyqat/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_pol_privacyqa () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pol \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pol/privacyqa/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_pol_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pol \
    --do_eval \
    --do_lower_case \
    --predict_file $SQUAD_DIR/eval.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pol/squad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_polsquad_policyqad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $polsquad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/dev.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$polsquad/policyqad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_polsquad_policyqat () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $polsquad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$polsquad/policyqat/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_polsquad_privacyqa () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $polsquad \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$polsquad/privacyqa/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_polsquad_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $polsquad \
    --do_eval \
    --do_lower_case \
    --predict_file $SQUAD_DIR/eval.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$polsquad/squad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

# PrivacyQA Evals

function evaluate_pri_policyqad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pri \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/dev.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pri/policyqad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_pri_policyqat () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pri \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pri/policyqat/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_pri_privacyqa () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pri \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pri/privacyqa/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_pri_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $pri \
    --do_eval \
    --do_lower_case \
    --predict_file $SQUAD_DIR/eval.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$pri/squad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_prisquad_policyqad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $prisquad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/dev.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$prisquad/policyqad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_prisquad_policyqat () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $prisquad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$prisquad/policyqat/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_prisquad_privacyqa () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $prisquad \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$prisquad/privacyqa/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_prisquad_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $prisquad \
    --do_eval \
    --do_lower_case \
    --predict_file $SQUAD_DIR/eval.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$prisquad/squad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

# SQUAD Evals

function evaluate_squad_policyqad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/dev.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$squad/policyqad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

function evaluate_squad_policyqat () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_eval \
    --do_lower_case \
    --predict_file $POLICYQA_DIR/test.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$squad/policyqat/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_squad_privacyqa () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_eval \
    --do_lower_case \
    --predict_file $PRIVACYQA_DIR/policy_test_squad.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$squad/privacyqa/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}


function evaluate_squad_squad () {

CUDA_VISIBLE_DEVICES=0 python run_squad.py \
    --model_type bert \
    --model_name_or_path $base_squad \
    --do_eval \
    --do_lower_case \
    --predict_file $SQUAD_DIR/eval.json \
    --max_seq_length $max_seq_length \
    --doc_stride $doc_stride \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir ./$squad/squad/ \
    --per_gpu_eval_batch_size=$eval_batch_size
}

rm -f cached_*

evaluate_pol_policyqad
evaluate_pol_policyqat
evaluate_pol_privacyqa
evaluate_pol_squad

evaluate_polsquad_policyqad
evaluate_polsquad_policyqat
evaluate_polsquad_privacyqa
evaluate_polsquad_squad

evaluate_pri_policyqad
evaluate_pri_policyqat
evaluate_pri_privacyqa
evaluate_pri_squad

evaluate_prisquad_policyqad
evaluate_prisquad_policyqat
evaluate_prisquad_privacyqa
evaluate_prisquad_squad

evaluate_squad_policyqad
evaluate_squad_policyqat
evaluate_squad_privacyqa
evaluate_squad_squad
