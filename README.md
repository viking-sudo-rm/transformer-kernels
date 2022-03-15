# Transformer Kernels

## Transformers With/Without Keys and Queries

This experiment compares transformer language models with identity keys/queries to normal transformer language models.

Dependencies include allennlp:
``shell
pip install allennlp allennlp-models
```

To train a transformer language model:

```shell
export CUDA=0
ID=1 python -m allennlp train training_config/transformer.jsonnet -s=models/identity --include-package=src
ID=0 python -m allennlp train training_config/transformer.jsonnet -s=models/no-identity --include-package=src
```