4 Experiment
4.1 Experimental Setup
Models and Datasets In our experiments, we use variants of the DeepSeek-R1 distilled model:
DeepSeek-R1-Distill-Llama-8B, and DeepSeek-R1-Distill-Qwen-14B [1], which we refer to as
R1-Llama-8B and R1-Qwen-14B, respectively, for brevity throughout the paper.
We evaluate the models’ mathematical reasoning capabilities using three benchmarks: MATH-500
[8] and AIME 2024 [9].
Hyperparameters We set Bbuffer = 128, α= 8 and λ= 0.1, with an analysis of λin §5.1.
Baselines We compare our method against SnapKV [3], originally designed for long prefilling.
To adapt it for decoding, we apply the same compression interval as our method, i.e., compressing
the KV cache every 128 decoding steps using identical Bbudget and Bbuffer. Our approach focuses
on improving KV cache eviction through a hybrid strategy, and we therefore restrict comparison to
state-of-the-art attention-based eviction methods. Budget allocation techniques (e.g., head-level [6]
and layer-level [5]) are orthogonal to our work and not included. We also report results for FullKV,
which retains the full KV cache and serves as the gold standard for decoding quality.
Evaluation Setup We set the maximum generation length to 16,384 tokens for MATH-500 and
32,768 tokens for AIME 2024 and AIME 2025, because further increasing the generation length
has shown no improvement on model performance on these datasets from our attempts. We find
that using greedy decoding to evaluate long-output reasoning models results in significant variability
across different setups. Following existing works [1], we utilize pass@kevaluation [10] and report
pass@1 using a non-zero temperature. We use the recommended sampling temperature and top-p
value for each model, i.e., sampling temperature of 0.6 and a top-pvalue of 0.95 for DeepSeek-
R1 Distilled models. We generate 64 responses for each question. Pass@1 is then calculated as
Pass@1=
1
k
k
i=1 pi (这里这个公式显示的有问题 我看了一下大概的意思就是一道题的平均正确率，每个回答就是对或者是错，对应着 0/1), where pi denotes the correctness of the i-th response. This method provides
more reliable performance estimates.
The accuracy performance of R-KV compared with all baselines is shown in Figure 4, with detailed
accuracy numbers in Appendix B.2. The KV cache budget ratio is calculated based on the KV cache
budget and the average generation length of tokens, i.e., R1-Llama-8B: 2,979.1 on MATH-500
and 15,535.8 on AIME24; R1-Qwen-14B: 2,833.04 on MATH-500 and 12,402 on AIME24. Our
method significantly outperforms the baseline SnapKV, achieving up to 40% Acc. improvement. We
provide two KV cache budget and performance analysis. Fixed budget analysis is more practical
because when the model outputs longer (i.e., from 2,979.1 on MATH-500 to 15,535.8 on AIME24),
the KV cache budget needed for lossless compression increases less (i.e., 512). In the KV cache
budget ratio perspective, the changes of lossless compression ratio is dominated by generation length.
Ratio Budget For R1-Llama-8B, R-KV achieves lossless compression with 34% KV cache budget
on the MATH-500 dataset and with 10% KV cache budget on the AIME-2024 dataset. Given 16%
KV cache budget, our method even surpasses the FullKV baseline, reaching 105% of its accuracy.
Similarly, for R1-Qwen-14B, R-KV achieves lossless compression with 54% KV cache budget on
the MATH-500 dataset and with 25% KV cache budget on the AIME-2024 dataset. Given 33% KV
cache budget, our method achieves 105% of FullKV accuracy.
Fixed Budget For R1-Llama-8B, R-KV achieves lossless compression with 1024 KV cache budget
on the MATH-500 dataset and with 1536 KV cache budget on the AIME-2024 dataset. For R1-Llama-
8B, R-KV achieves lossless compression with 1536 KV cache budget on the MATH-500 dataset and
with 3072 KV cache budget on AIME-2024.