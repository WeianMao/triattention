BENCHMARK="math" # "gsm8k", "math"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  
MODEL_SIZE="7b"  
MODEL_TYPE="qwen" # "llama3", "qwen"
DATA_TYPE="test" 

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=8192 # qwen: 4096 for gsm8k, 8192 for math, 16384 for aime
EVAL_BATCH_SIZE=1 
TEMPERATURE=0.0
SEED=42

#KV Compression Settings
method=Window_LAZY # Support FullKV, Window_LAZY
max_kv_capacity=1492   
attn_implementation="sdpa" 
decoding_recent_size=175   # gsm 101; math 175; aime 363

OUPTUT_DIR="outputs/DeepSeek-R1-Distill-Qwen-7B/${BENCHMARK}"  

CUDA_VISIBLE_DEVICES=0 python ./evaluation.py --output-dir ${OUPTUT_DIR} --model-path ${MODEL_PATH} --tokenizer-path ${MODEL_PATH} \
    --model-size ${MODEL_SIZE} --model-type ${MODEL_TYPE} --data-type ${DATA_TYPE}  --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} --eval_batch_size ${EVAL_BATCH_SIZE} --temperature ${TEMPERATURE} --seed ${SEED} --benchmark ${BENCHMARK}  \
    --method ${method} \
    --use_cache True \
    --max_kv_capacity ${max_kv_capacity} \
    --decoding_recent_size ${decoding_recent_size} \
    --attn_implementation ${attn_implementation} \
    --alpha 0.0001 