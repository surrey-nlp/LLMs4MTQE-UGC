# LLMs4MTQE-UGC
This repository contains the code and data for LLMs to evaluate machine translation of emotion-loaded Chinese user-generated content (UGC). Our paper "Are Large Language Models State-of-the-art Quality Estimators for Machine Translation of User-generated Content?" will be presented at the 11th Workshop of Asian Translation at EMNLP2024. The paper investigates whether LLMs are better quality estimators than fine-tuning of multilingual pre-trained language models under in-context learning (ICL) and parameter-efficient fine-tuning (PEFT) scenarios. For more details, please find our arXiv preprint.

## Installation

We used LLaMA-Factory (Zheng et al., 2024) for both ICL and PEFT. You can follow our instructions below to install LLaMA-Factory or refer to their [GitHub](https://github.com/hiyouga/LLaMA-Factory).

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Data

The data for this paper is under the "data" folder, which is specifically formatted for LLM training and inference using LLaMA-Factory. The orginal dataset was proposed in [our paper](https://aclanthology.org/2023.eamt-1.13/) and can be found at our [GitHub repository](https://github.com/surrey-nlp/HADQAET). 

After downloading all data files under our "data" folder, copy all data files to the "data" subfolder under "LLaMA-Factory". Then, open the "dataset_info.json" file and add the following into the file to prepare for LLM inference or PEFT.

```
  "train_p1": {
    "file_name": "train_p1.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "train_p2": {
    "file_name": "train_p2.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "test_p1": {
    "file_name": "test_p1.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "test_p2": {
    "file_name": "test_p2.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "test_p1fsl": {
    "file_name": "test_p1fsl.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "test_p2fsl": {
    "file_name": "test_p2fsl.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
```

## PEFT

You can follow our instructions below for PEFT of LLMs and refer to their [GitHub](https://github.com/hiyouga/LLaMA-Factory) for more details.

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --do_train \
    --dataset train_p1 \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir outputs/llama2_p1 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --fp16 \
    --lora_rank 8
```

## Inference

You can follow our instructions below for LLM inference and refer to their [GitHub](https://github.com/hiyouga/LLaMA-Factory) for more details.

```
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --adapter_name_or_path outputs/llama2_p1 \
    --quantization_bit 4 \
    --dataset test_p1 \
    --template llama2 \
    --finetuning_type lora \
    --output_dir preds \
    --per_device_eval_batch_size 8 \
    --max_samples 10000 \
    --predict_with_generate \
    --fp16 \
    --max_length 2048 \
    --max_new_tokens 1024
```

## References

Yaowei Zheng, Richong Zhang, Junhao Zhang, YeYanhan YeYanhan, and Zheyan Luo. 2024. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, pages 400–410, Bangkok, Thailand. Association for Computational Linguistics.

Shenbin Qian, Constantin Orasan, Felix Do Carmo, Qiuliang Li, and Diptesh Kanojia. 2023. Evaluation of Chinese-English Machine Translation of Emotion-Loaded Microblog Texts: A Human Annotated Dataset for the Quality Assessment of Emotion Translation. In *Proceedings of the 24th Annual Conference of the European Association for Machine Translation*, pages 125–135, Tampere, Finland. European Association for Machine Translation.