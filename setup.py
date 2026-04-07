from setuptools import setup, find_packages

setup(
    name="triattention",
    version="0.1.0",
    description="TriAttention: efficient KV cache compression via tri-directional sparse attention",
    author="Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen",
    url="https://github.com/WeianMao/triattention",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.48.1",
        "datasets>=4.0",
        "huggingface-hub>=0.35",
        "accelerate",
        "numpy>=1.26",
        "scipy",
        "einops",
        "sentencepiece",
        "pyyaml>=6.0",
        "tqdm",
        "matplotlib",
        "regex",
        "torch",
        "triton",
    ],
    extras_require={
        "eval": [
            "pebble>=5.0",
            "sympy>=1.13",
            "latex2sympy2",
            "word2number",
            "antlr4-python3-runtime==4.7.2",
        ],
        "flash": ["flash-attn>=2.5.8"],
    },
    entry_points={
        "vllm.general_plugins": [
            "triattention = triattention.vllm.plugin:register_triattention_backend",
        ],
    },
)
