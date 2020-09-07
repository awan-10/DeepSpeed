---
title: "1-bit Adam"
excerpt: ""
---

In this tutorial we are going to introduce 1-bit Adam in Deepspeed. We start with the example of using 1-bit Adam for BingBertSquad training, and then BERT Pretraining.

## Overview

If you don't already have a copy of the DeepSpeed repository, please clone in
now and checkout the DeepSpeedExamples submodule the contains the BingBertSquad
example (DeepSpeedExamples/BingBertSquad) we will be going over in the rest of
this tutorial.

```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git submodule update --init --recursive
cd DeepSpeedExamples/BingBertSquad
```
##1-bit Adam for BingBertSquad
### Pre-requisites

* Download SQuAD data:
  * Training set: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
  * Validation set: [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

You also need a pre-trained BERT model checkpoint from either DeepSpeed, [HuggingFace](https://github.com/huggingface/transformers), or [TensorFlow](https://github.com/google-research/bert#pre-trained-models) to run the fine-tuning. Regarding the DeepSpeed model, we will use checkpoint 160 from the BERT pre-training [tutorial](/tutorials/bert-pretraining/).

### Running BingBertSquad

- **DeepSpeed-enabled:** We provide a shell script that you can invoke to start training with DeepSpeed, it takes 4 arguments: `bash run_squad_deepspeed.sh <NUM_GPUS> <PATH_TO_CHECKPOINT> <PATH_TO_DATA_DIR> <PATH_TO_OUTPUT_DIR>`. The first argument is the number of GPUs to train with, second argument is the path to the pre-training checkpoint, third is the path to training and validation sets (e.g., train-v1.1.json), and fourth is path to an output folder where the results will be saved. This script will invoke `nvidia_run_squad_deepspeed.py`.


## DeepSpeed Integration

The main part of training is done in `nvidia_run_squad_deepspeed.py`, which has
already been modified to use DeepSpeed. The `run_squad_deepspeed.sh` script
helps to invoke training and setup several different hyperparameters relevant
to the training process. In the next few sections we will cover what changes we
made to the baseline in order to enable DeepSpeed, you don't have to make these
changes yourself since we have already done them for you.

### Launching 
To enable the 1-bit compressed training, 1-bit Adam uses [xxx] as the communication backend, which means we use MPI (e.g., mpirun) as the launcher. With [cuda_aware], the training can be launched by using:
```shell
mpirun -np [#processes] -ppn [#GPUs on each node] -hostfile [hostfile] ... bash mpi_run_squad_deepspeed.sh
```
For clusters where [cuda_aware=False], please run the following command for launching:
```shell
mpirun -np [#processes] -npernode [#GPUs on each node] -hostfile [hostfile] ... bash mpi_run_squad_deepspeed.sh
```

### Configuration

The `deepspeed_bsz96_onebit_config.json` file gives the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters.
When running the `nvidia_run_squad_deepspeed.py`, in addition to the
`--deepspeed` flag to enable DeepSpeed, the appropriate DeepSpeed configuration
file must be specified using `--deepspeed_config
deepspeed_bsz24_config.json`. Table 1 shows the fine-tuning configuration
used in our experiments.

| Parameters                     | Value |
| ------------------------------ | ----- |
| Total batch size               | 96    |
| Train micro batch size per GPU | 3     |
| Optimizer                      | **OnebitAdam**  |
| Learning rate                  | 3e-5  |
| Sequence-length                | 384   |
| Weight-decay                   | 0.0   |
| Epoch count                    | 2     |
| **freeze_step**                | 400     |
| **cuda_aware**                    | True     |
Table 1. Fine-tuning configuration

Notice that for 1-bit Adam, the *freeze_step* controls number of the optimizer step for running the original uncompressed Adam, and after that, the training will be using 1-bit compression for communication. *cuda_aware* (default as True) is a flag to control whether to [xxx].


### Training
The arguement parsing, initialization, forward pass, backward pass, weight update and evaluation are similar to the BingBertSquad tutorial. [link to that tutorial]


### Fine-tuning Results
The table summarizing the results are given below. In all cases (unless
otherwise noted), the total batch size is set to 24 and training is conducted
on 4 GPUs for 2 epochs on a DGX-2 node.  A set of parameters (seeds and
learning rates) were tried and the best ones were selected. All learning rates
were 3e-5; We set the seeds to 9041 and 19068 for HuggingFace and TensorFlow
models, respectively. The checkpoints used for each case are linked in the
table below.

| Case        | Model                                 | Precision | EM    | F1    |
| ----------- | ------------------------------------- | --------- | ----- | ----- |
| HuggingFace | [Bert-large-uncased-whole-word-masking](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin) | FP16      | 87.26 | 93.32 |



### Loading HuggingFace and TensorFlow Pretrained Models

BingBertSquad supports both HuggingFace and TensorFlow pretrained models. Here,
we show the two model examples:

1. `test/huggingface` which includes the checkpoint
[Bert-large-uncased-whole-word-masking](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin) and [bert json config](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json).
2. `test/tensorflow` which comes from a checkpoint zip from Google
[Bert-large-uncased-L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip).

```shell
[test/huggingface]
bert-large-uncased-whole-word-masking-config.json
bert-large-uncased-whole-word-masking-pytorch_model.bin
```

```shell
[test/tensorflow]
bert_config.json
bert_model.ckpt.data-00000-of-00001
bert_model.ckpt.index
bert_model.ckpt.meta
```

There are three arguments used for loading these two types of checkpoints.

1. `--model_file`, points to the pretrained model file.
2. `--ckpt_type`, indicates the checkpoint type, `TF` for Tensorflow, `HF` for HuggingFace, default value is `DS` for DeepSpeed.
3. `--origin_bert_config_file`, points to the BERT config file, usually saved in same folder of `model_file`.

We can add the following in our fine-tuning shell script in
`run_squad_deepspeed.sh` to run the above HuggingFace and TensorFlow examples.

```shell
[HuggingFace]

--model_file test/huggingface/bert-large-uncased-whole-word-masking-pytorch_model.bin \
--ckpt_type HF \
--origin_bert_config_file test/huggingface/bert-large-uncased-whole-word-masking-config.json \
```

```shell
[TensorFlow]

--model_file /test/tensorflow/bert_model.ckpt \
--ckpt_type TF \
--origin_bert_config_file /test/tensorflow/bert_config.json \
```

Note:

1. `--deepspeed_transformer_kernel` flag is required for using HuggingFace or TensorFlow pretrained models.

2. `--preln` flag cannot be used with HuggingFace or TensorFlow pretrained models, since they use a post-layer-norm.

3. BingBertSquad will check the pretrained models to have the same vocabulary size and won't be able to run if there is any mismatch. We advise that you use a model checkpoint of the style described above or a DeepSpeed bing\_bert checkpoint.

### Tuning Performance
In order to perform fine-tuning, we set the total batch size to 24 as shown in Table 1. However, we can tune the micro-batch size per GPU to get high-performance training. In this regard, we have tried different micro-batch sizes on NVIDIA V100 using either 16GB or 32GB of memory. As Tables 2 and 3 show, we can improve performance by increasing the micro-batch. Compared with PyTorch, we can achieve up to 1.5x speedup for the 16GB V100 while supporting a 2x larger batch size per GPU. On the other hand, we can support as large as 32 batch size (2.6x higher than PyTorch) using a 32GB V100, while providing 1.3x speedup in the end-to-end fine-tune training. Note, that we use the best samples-per-second to compute speedup for the cases that PyTorch runs out-of-memory (OOM).

| Micro batch size | PyTorch | DeepSpeed | Speedup (x) |
| ---------------- | ------- | --------- | ----------- |
| 4                | 36.34   | 50.76     | 1.4         |
| 6                | OOM     | 54.28     | 1.5         |
| 8                | OOM     | 54.16     | 1.5         |

Table 2. Samples/second for running SQuAD fine-tuning on NVIDIA V100 (16GB) using PyTorch and DeepSpeed transformer kernels.

| Micro batch size | PyTorch | DeepSpeed | Speedup (x) |
| ---------------- | ------- | --------- | ----------- |
| 4                | 37.78   | 50.82     | 1.3         |
| 6                | 43.81   | 55.97     | 1.3         |
| 12               | 49.32   | 61.41     | 1.2         |
| 24               | OOM     | 60.70     | 1.2         |
| 32               | OOM     | 63.01     | 1.3         |

Table 3. Samples/second for running SQuAD fine-tuning on NVIDIA V100 (32GB) using PyTorch and DeepSpeed transformer kernels.

As mentioned, we can increase the micro-batch size per GPU from 3 to 24 or even
higher if a larger batch size is desired. In order to support a larger
micro-batch size, we may need to enable different memory-optimization flags for our
transformer kernel as described in [DeepSpeed Transformer
Kernel](/tutorials/transformer_kernel/) tutorial. Table 4 shows which
optimization flags are required for running different range of micro-batch
sizes.

| Micro batch size |           NVIDIA V100 (32-GB)            |           NVIDIA V100 (16-GB)            |
| :--------------: | :--------------------------------------: | :--------------------------------------: |
|       > 4        |                    -                     |          `normalize_invertible`          |
|       > 6        |                    -                     | `attn_dropout_checkpoint`, `gelu_checkpoint` |
|       > 12       | `normalize_invertible`, `attn_dropout_checkpoint` |                   OOM                    |
|       > 24       |            `gelu_checkpoint`             |                   OOM                    |

Table 4. The setting of memory-optimization flags for a range of micro-batch size on 16-GB and 32-GB V100.

### FineTuning model pre-trained with DeepSpeed Transformer Kernels

Fine-tuning the model pre-trained using DeepSpeed Transformer and the recipe in [DeepSpeed Fast-Bert Training](https://www.deepspeed.ai/news/2020/05/27/fastest-bert-training.html) should yield F1 score of 90.5 and is expected to increase if you let the pre-training longer than suggested in the tutorial.

To get these results, we do require some tuning of the dropout settings as described below:

### Dropout Setting
For the fine-tuning, we only use the deterministic transformer to have reproducible the fine-tuning results. But, we choose different values for dropout based on whether pre-training was done using deterministic or stochastic transformer (Please see [Transformer tutorial](/tutorials/transformer_kernel/) for more detail of selecting these two modes).

For models pre-trained with deterministic transformer, we use the same dropout ratio used in pre-training (0.1). However, we slightly increase the dropout ratio when fine-tuning the model pre-trained using the stochastic transformer to compensate for the lack of stochastic noise during fine-tuning.


| Pre-training mode | Dropout ratio |
| ----------------- | ------------- |
| Deterministic     | 0.1           |
| Stochastic        | 0.12 - 0.14   |
