from setuptools import setup, find_packages

setup(
    name="triattention",
    version="0.1.0",
    description="TriAttention: efficient KV cache compression via tri-directional sparse attention",
    author="Anonymous",
    url="https://github.com/placeholder/triattention",
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
    ],
)
