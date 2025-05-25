## 1. About megatron

+ 💻**source code**: https://github.com/NVIDIA/Megatron-LM
+ 📰**paper:** https://arxiv.org/pdf/2406.07887
+ 📽️ [李沐论文精读](https://www.bilibili.com/video/BV1nB4y1R7Yz/?vd_source=2b4793ad721ff6ac59256683a01dd0c0)

## 2. Usage step
### 2.1 installation

We strongly recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)[^1] with DGX nodes. If you can't use this for some reason, use the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases. Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

You can launch an instance of the PyTorch container and mount Megatron, your dataset, and checkpoints with the following Docker commands:

```shell
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/megatron:/workspace/megatron -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints nvcr.io/nvidia/pytorch:xx.xx-py3
```

### 2.2 Download Checkpoint(if need)

We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) and [GPT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) checkpoints to evaluate or for finetuning downstream tasks. To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:
``` shell
# BERT-345M-uncased: 
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
# BERT-345M-cased: 
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
# GPT-345M: 
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
```

The models require vocabulary files to run. The BERT WordPiece vocab file can be extracted from Google's pretrained BERT models: [uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.
### 2.3 train the model

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation
However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

#### 2.3.1 Data preprocessing

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
``` text
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for BERT training is:
``` shell
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
```

**The output will be two files named**, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`.

#### 2.3.2 Pretrain

+ **How to run?** Run `pretrain_{bert,gpt,t5}_distributed.sh`

> [!NOTE] 💡how to run distributed training? 
> scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables. See the official PyTorch [documentation](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization).

+ **Parallelism:** We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`.
	+ ***DP***: use pytorch interface.
	+ ***Model parallelism***:  2 level (TP & SP)
	+ ***Pipeline parallelism***: ❓key is to split the dataset and control the training progress?

> [!Note] 💡Use of TP & SP
> Second, we developed a simple and efficient two-dimensional model-parallel approach. To use the first dimension, tensor model parallelism (splitting execution of a single transformer module over multiple GPUs, see Section 3 of [our paper](https://arxiv.org/pdf/1909.08053.pdf)), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use the second dimension, sequence parallelism, specify `--sequence-parallel`, which also requires tensor model parallelism to be enabled because it splits across the same GPUs (more details in Section 4.2.2 of [our paper](https://arxiv.org/pdf/2205.05198.pdf)).

> [!NOTE] 💡Use of PP
>To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches, see Section 2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).
>
>The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num-layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).


+ **Overlap backward and gradient reduction**: with the backward pass when the `--overlap-grad-reduce` command-line option is used.

+ **Activation Checkpointing and Recomputation**：
	+ We support two levels of recompute granularity:  `selective` and `full`
		+ 🛣️**selective**: retains in memory the activations that take less memory storage space and are more expensive to recompute. recomputes the activations that take more memory storage space but are relatively inexpensive to recompute. `--recompute-activations` to use.
		+ 🌕**full**：recompute saves just the inputs to a transformer layer, or a group, or block, of transformer layers, and recomputes everything else. `--recompute-granularity full`.
			+ When using `full` activation recompute, there are two methods: `uniform` and `block`, chosen using the `--recompute-method` argument.
			+ **The `uniform` method** uniformly divides the transformer layers into groups of layers (each group of size `--recompute-num-layers`) and stores the input activations of each group in memory. The baseline group size is 1 and, in this case, the input activation of each transformer layer is stored. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage, enabling a bigger model to be trained. For example, when `--recompute-num-layers` is set to 4, only the input activation of each group of 4 transformer layers is stored.
			+ **The `block` method** : Base on PP(pipeline paralism), For example, when we specify 5 layers(by `--recompute-num-layers`) to recompute of 8 layers per pipeline stage, the input activations of only the first 5 transformer layers are recomputed in the backprop step while the input activations for the final 3 layers are stored.
+ **Distributed Optimizer**
	+ Usage: `--use-distributed-optimizer`. Compatible with all model and data types.
	+ The distributed optimizer is a memory savings technique, whereby the optimizer state is evenly distributed across data parallel ranks (versus the traditional method of replicating the optimizer state across data parallel ranks). As described in [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
+ **Flash Attention**
	+ Usage: `--use-flash-attn`. Support attention head dimensions at most 128.
	+ [FlashAttention](https://github.com/HazyResearch/flash-attention) is a fast and memory-efficient algorithm to compute exact attention. It speeds up model training and reduces memory requirement.
		To install FlashAttention: `pip install flash-attn`
+ **MOE**
	+ **Introduction:** MoE (Mixture of Experts) is a powerful LLM architecture implemented in the Megatron-Core framework, designed to enhance the efficiency and scalability of large language models. It leverages **Expert Parallelism**, allowing multiple experts to be distributed across different workers, where each worker processes distinct batches of training samples. This method significantly increases computational throughput, enabling models to achieve high performance metrics, such as 47% MFU during BF16 training for 8x7B on H100.
	+ **Key Features of MoE:**
		- **Parallelism Techniques**: MoE combines various parallelism strategies, including Expert Parallelism, Data Parallelism, Tensor Parallelism, Sequence Paralleism, Pipeline Parallelism, and Context Parallelism. This combination allows for handling larger model variants effectively.
		- **Router and Load Balancing**: The system employs advanced routing mechanisms like the Top-K router and utilizes load balancing algorithms to optimize token distribution among experts.
		- **Performance Optimizations**: Techniques such as GroupedGEMM and FP8 training enhance the efficiency of MoE models, particularly when multiple experts are involved.
		- **Token Dispatch Mechanism**: MoE supports both dropless and token drop strategies to manage token distribution effectively across experts.
[^1]: NGC: Nvidia GPU Cloud
