#!/bin/bash
python evaluation/eval_math.py \
    --exp_name "evaluation" \
    --output_dir "." \
    --base_dir "./outputs" \
    --dataset aime24
