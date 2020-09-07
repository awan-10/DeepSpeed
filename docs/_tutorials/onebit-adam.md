---
title: "1-bit Adam"
---

In this tutorial we are going to introduce 1-bit Adam in Deepspeed, which could potentially improve the training speed in a communication intensive scenario by reducing the communication vlolume. To see how to use 1-bit Adam in DeepSpeed, we use the following two training tasks as example:  

* [BingBertSQuAD Fine-tuning](/tutorials/bert-finetuning/)
* [BERT Pre-training](/tutorials/bert-pretraining/)

For more details, please refer to the [BingBertSQuAD Fine-tuning](/tutorials/bert-finetuning/) and [BERT Pre-training](/tutorials/bert-pretraining/) posts.
## Overview

If you don't already have a copy of the DeepSpeed repository, please clone in
now and checkout the DeepSpeedExamples submodule the contains the BingBertSQuAD
example (DeepSpeedExamples/BingBertSQuAD) we will be going over in the rest of
this tutorial.

```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git submodule update --init --recursive
cd DeepSpeedExamples/BingBertSQuAD
```
##1-bit Adam for BingBertSQuAD
### Pre-requisites

* Download SQuAD data:
  * Training set: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
  * Validation set: [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

You also need a pre-trained BERT model checkpoint from either DeepSpeed, [HuggingFace](https://github.com/huggingface/transformers), or [TensorFlow](https://github.com/google-research/bert#pre-trained-models) to run the fine-tuning. Regarding the DeepSpeed model, we will use checkpoint the checkpoint from [Hugging Face](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin).

### Running BingBertSQuAD

- **DeepSpeed-enabled:** We provide a shell script that you can invoke to start training with DeepSpeed, it takes 4 arguments: `bash run_squad_deepspeed.sh <NUM_GPUS> <PATH_TO_CHECKPOINT> <PATH_TO_DATA_DIR> <PATH_TO_OUTPUT_DIR>`. The first argument is the number of GPUs to train with, second argument is the path to the pre-training checkpoint, third is the path to training and validation sets (e.g., train-v1.1.json), and fourth is path to an output folder where the results will be saved. This script will invoke `nvidia_run_squad_deepspeed.py`.


## DeepSpeed Integration

The main part of training is done in `nvidia_run_squad_deepspeed.py`, which has
already been modified to use DeepSpeed. The `run_squad_deepspeed.sh` script
helps to invoke training and setup several different hyperparameters relevant
to the training process. 



### Configuration

The `deepspeed_bsz96_onebit_config.json` file gives the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters.
When running the `nvidia_run_squad_deepspeed.py`, in addition to the
`--deepspeed` flag to enable DeepSpeed, the appropriate DeepSpeed configuration
file must be specified using `--deepspeed_config
deepspeed_bsz96_config.json`. Table 1 shows the fine-tuning configuration
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

### Launching 
To enable the 1-bit compressed training, 1-bit Adam uses [xxx] as the communication backend, which means we use MPI (e.g., mpirun) as the launcher. With [xxx], the training can be launched by using:
```shell
mpirun -np [#processes] -ppn [#GPUs on each node] -hostfile [hostfile] [MPI flags] bash mpi_run_squad_deepspeed.sh
```
For example, in order to use 32 GPUs (4GPUs/node, 8 nodes in total), with the support of InfiniBand, you can use Mvapich2 as the launcher and run the following command:
```shell
mpirun -np 32 -ppn 4 -hostfile hosts MV2_USE_GDRCOPY=0 -env MV2_SMP_USE_CMA=0 -env MV2_DEBUG_SHOW_BACKTRACE=1 -env MV2_USE_CUDA=1 -env MV2_SUPPORT_DL=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_INTER_ALLGATHER_TUNING=5 -env MV2_CUDA_USE_NAIVE=0 bash run_squad_deepspeed.sh
```
For clusters where [xxx], please run the following command for launching:
```shell
mpirun -np [#processes] -npernode [#GPUs on each node] -hostfile [hostfile] [MPI flags] bash mpi_run_squad_deepspeed.sh
```
Similarly, to use 32 GPUs (4GPUs/node, 8 nodes in total) in a TCP communication system, you can run
 ```shell
mpirun -np 32 -npernode 8 -hostfile hosts [Ammar: add flag thx] bash mpi_run_squad_deepspeed.sh
```


### Training
For more details about loading checkpoint, arguement parsing, initialization, forward pass, backward pass, weight update and evaluation,please refer to the [BingBertSQuAD Fine-tuning](/tutorials/bert-finetuning/) tutorial. 


### Fine-tuning Results
The table summarizing the results are given below. In all cases (unless
otherwise noted), the total batch size is set to 96 and training is conducted
on 32 GPUs for 2 epochs on a DGX-2 node.  A set of parameters (seeds and
learning rates) were tried and the best ones were selected. All learning rates
were 3e-5. The checkpoints used for each case are linked in the
table below.

| Case        | Model                                 | Precision | EM    | F1    |
| ----------- | ------------------------------------- | --------- | ----- | ----- |
| HuggingFace | [Bert-large-uncased-whole-word-masking](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin) | FP16      | 87.26 | 93.32 |


##1-bit Adam for BERT Pre-training
### Pre-requisites
Please refer to [BERT Pre-training](/tutorials/bert-pretraining/) for more details about data downloading and pre-processing.



