DestroyGPU
opened on Jul 16
Hi! Thank you for your hard work!

When I directly ran the script you provided in the scripts folder, I encountered the following error:

assert budget - window_size > 0, "budget must be greater than window_size"
TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
To address this, I revised the script and ran the following commands:

export CUDA_VISIBLE_DEVICES=7

python3 ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./outputs/output.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
--max_length 32768 \
--eval_batch_size 1 \
--method rkv \
--kv_budget 2048 \
--window_size 8 \
--retain_ratio 0.2 \
--mix_lambda 0.1
The output I received was:

{'num_samples': 30, 'num_scores': 30, 'timeout_samples': 0, 'empty_samples': 0, 'acc': np.float64(36.7)}
Saved to ./evaluation/aime24/default-aime24_math_eval.jsonl
The accuracy is 36.7%, which is significantly lower than the >50% result reported in the paper.
Could you please confirm whether the configuration I used conform with the experiment in your paper and help me understand why the accuracy is so much lower than expected?

Thank you!!!
Activity
TTTTTTris
TTTTTTris commented on Jul 22
TTTTTTris
on Jul 22
I encountered the same issue and would appreciate any guidance from the authors.
Zefan-Cai
Zefan-Cai commented on Jul 25
Zefan-Cai
on Jul 25
Owner
The revision of the script looks good. To reproduce the result, you would need to sample 64 times and calculate the average score.
Xiangyi1996
Xiangyi1996 commented on Aug 18
Xiangyi1996
on Aug 18 · edited by Xiangyi1996
Hi, I'm attempting to implement this correctly but would like to confirm my understanding:

For "sample 64 times", should I modify the inference code to i.e.:do_sample=True,temperature=0.6, top_p=0.95 num_return_sequences=64 in model.generate for the sampling variations?

If my understanding is incorrect, could you please clarify the proper implementation approach? Any code snippet examples would be greatly appreciated.
Zefan-Cai
Zefan-Cai commented on Aug 19
Zefan-Cai
on Aug 19
Owner
I would clarify the experiments recently and provide detailed output jsons, benchmark scores and config for all of you. In fact our results were based on a self-bulit inference infra based on Flash-Infer. I am not very familiar with HuggingFace behaviors. I would try my best to help.
Zefan-Cai
Zefan-Cai commented on Aug 19
Zefan-Cai
on Aug 19
Owner
Hi, I'm attempting to implement this correctly but would like to confirm my understanding:

For "sample 64 times", should I modify the inference code to i.e.:do_sample=True,temperature=0.6, top_p=0.95 num_return_sequences=64 in model.generate for the sampling variations?

If my understanding is incorrect, could you please clarify the proper implementation approach? Any code snippet examples would be greatly appreciated.
The configs lokk good.
TTTTTTris
TTTTTTris commented on Aug 19
TTTTTTris
on Aug 19
May I ask what eval_batch_size did you use to get the accuracy results in the paper? I found that using a larger eval_batch_size degrades the performance a lot. Not sure if I have a wrong implementation cause the original code seems do not support multi-batch inference. Thanks :)
Zefan-Cai
Zefan-Cai commented on Aug 19
Zefan-Cai
on Aug 19
Owner
May I ask what eval_batch_size did you use to get the accuracy results in the paper? I found that using a larger eval_batch_size degrades the performance a lot. Not sure if I have a wrong implementation cause the original code seems do not support multi-batch inference. Thanks :)
The bachify in the code has problem. I would fix it recently. Currently, use eval_batch_size=1 can reproduce the results.
lsjia
lsjia commented on Sep 18
lsjia
on Sep 18
May I ask what eval_batch_size did you use to get the accuracy results in the paper? I found that using a larger eval_batch_size degrades the performance a lot. Not sure if I have a wrong implementation cause the original code seems do not support multi-batch inference. Thanks :)
I met this problem too. When I set eval_batch_size>1, the performance of rkv and snapkv declines significantly. I think the possible reason is that the padding token was not handled correctly when calculating the attention scores.
Pwspang
Pwspang commented on Oct 7
Pwspang
on Oct 7
I would clarify the experiments recently and provide detailed output jsons, benchmark scores and config for all of you. In fact our results were based on a self-bulit inference infra based on Flash-Infer. I am not very familiar with HuggingFace behaviors. I would try my best to help.
Hi, do you have a link to the output JSONs, benchmark scores, and configuration files you mentioned? Having those would really help in reproducing the experiment. I ran the experiment on DeepSeek-R1-Distill-Qwen-7B, but observed that rkv has lower accuracy as compared to snapkv at lower budget.
lsjia
lsjia commented 3 weeks ago
lsjia
3 weeks ago
May I ask what eval_batch_size did you use to get the accuracy results in the paper? I found that using a larger eval_batch_size degrades the performance a lot. Not sure if I have a wrong implementation cause the original code seems do not support multi-batch inference. Thanks :)
The bachify in the code has problem. I would fix it recently. Currently, use eval_batch_size=1 can reproduce the results.
Hi, I wonder have you fixed this batchify problem? If so, can you update the repo? Thanks a lot.

aldinash
opened on Oct 16
Hello! In the paper it is mentioned that you do sampling with temp 0.6 and top_p 0.95 for all the evaluations. However, in run_math.py I see that

output = model.generate(
    **tokenized_prompts,
    max_length=args.max_length,
    do_sample=False,
    num_beams=1,
)
so there is no sampling. Could you please clarify this?
Activity
Zefan-Cai
Zefan-Cai commented on Oct 20
Zefan-Cai
on Oct 20
Owner
In the paper, we use sampling. Please use the setting described in the paper.