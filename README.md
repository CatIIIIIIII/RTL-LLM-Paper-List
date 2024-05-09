# LLM for RTL Debugging


## Table of Content (ToC)

- [LLM for RTL Debugging](#llm-for-rtl-debugging)
  - [Table of Content (ToC)](#table-of-content-toc)
  - [1. LLM Foundations](#1-llm-foundations)
  - [2. LLM for Coding](#2-llm-for-coding)
  - [3. LLM for RTL Coding](#3-llm-for-rtl-coding)

## 1. LLM Foundations

- [2024/01] **Mixtral of Experts** [[paper](https://arxiv.org/pdf/2401.04088)]  [[code](https://github.com/mistralai/mistral-src)]
  - One of the most popular open-source LLMs. A new model architecture that combines the strengths of transformer and mixture of experts.

- [2023/07] **Llama 2: Open Foundation and Fine-Tuned Chat Models** [[paper](https://arxiv.org/pdf/2307.09288)] [[code](https://github.com/facebookresearch/llama)]
  - IMPORTANT. The second version of LLaMA. It is used to initialize Code LLaMA.

- [2023/05] **REACT: Synergizing Reasoning and Acting in Language Models** [[paper](https://arxiv.org/pdf/2210.03629)] [[code](https://react-lm.github.io/)]
  - Reason, act step by step for LLM inference.

- [2023/05] **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT** [[paper](https://arxiv.org/pdf/2302.09419)] 
  - Review of pre-trained language models from BERT to ChatGPT.

- [2023/04] **Sparks of Artificial General Intelligence- Early experiments with GPT-4** [[paper](https://arxiv.org/pdf/2303.12712)] 
  - GPT-4 Guide book.

- [2023/02] **LLaMA: Open and Efficient Foundation Language Models** [[paper](https://arxiv.org/pdf/2302.13971)] [[code](https://github.com/facebookresearch/llama)]
  - Starter model of LLaMA family.

- [2023/01] **Secrets of RLHF in Large Language Models Part I** [[paper](https://arxiv.org/pdf/2307.04964)] [[code](https://github.com/OpenLMLab/MOSS-RLHF)]
  - Reinforcement human feedback for GPT, PPO strategy.

- [2022/12] **Scaling Instruction-Finetuned Language Models** [[paper](https://arxiv.org/pdf/2210.11416)] [[code](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
  - Instruction fine-tune also have “scaling” property. And when fine-tuning, need to add CoT dataset.

- [2022/02] **UL2: Unifying Language Learning Paradigms.** [paper](https://arxiv.org/pdf/2205.05131) [[code](https://github.com/google-research/google-research/tree/master/ul2)]
  - They propose a new Mixture of Denoisers (MoD) pretraining that frames multiple pretraining
tasks as span corruption, diversifies and then mixes them.

- [2022/02] **Towards A Unified View of Parameter-Efficient Transfer Learning** [[paper](https://arxiv.org/pdf/2202.08390)] [[code](https://github.com/jxhe/unify-parameter-efficient-tuning)]
  - IMPORTANT. The adaptation framework for LLMs.

- [2022/02] **Finetuned Language Models are Zero-shot Learners** [[paper](https://arxiv.org/pdf/2109.01652)] [[code](https://github.com/google-research/flan)]
  - IMPORTANT. They propose Instruction Fine-tuning, which is a new fine-tuning strategy for LLMs. It is widely used in the field of LLMs.
  
- [2021/10] **LORA: Low-Rank Adaptation of Large Language Models** [[paper](https://arxiv.org/pdf/2106.09685)] [[code](https://github.com/microsoft/LoRA)]
  - An efficient method to fine-tune LLMs.
  
- [2020/01] **Scaling laws for neural language models** [[paper](https://arxiv.org/pdf/2001.08361)] 
  - HARD and IMPORTANT. Investigate all kinds of power scaling of large language model based on transformer architecture.

- [2019/10] **Unified Language Model Pre-training for Natural Language Understanding and Generation** [[paper](https://arxiv.org/pdf/2005.14165)] [[code](https://github.com/microsoft/unilm)]
  - A mixed training strategy for both understand & generation task.

- [2019/10] **Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer** [[paper](https://arxiv.org/pdf/1910.10683)] [[code](https://github.com/google-research/text-to-text-transfer-transformer)]
  - IMPORTANT. Propose T5 which transform all NLP tasks into text-to-text format. The starter of prompt-based learning.
  
## 2. LLM for Coding

- [2024/04] **WizardCoder: Empowering Large Language Models to Follow Complex Instructions for Code Generation** [[paper](https://arxiv.org/pdf/2304.12244)] [[code](https://github.com/nlpxucan/WizardLM)]
  - IMPORTANT. Code model using evolution struct. Outperform GPT-4.

- [2024/01] **Code Llama: Open Foundation Models for Code** [[paper](https://arxiv.org/pdf/2308.12950)] [[code](https://github.com/facebookresearch/codellama)]
  - VERY IMPORTANT. Facebook's code LLM. We could follow their training pipeline to train our own code LLM. The most powerful open-source code LLM.
  
- [2024/01] **DebugBench: Evaluating Debugging Capability of Large Language Models** [[paper](https://arxiv.org/pdf/2401.04621)]  
  - A bench for code debugging. Use GPT to inplant bugs.
  
- [2024/01] **A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends** [[paper](https://arxiv.org/pdf/2311.10372)]
  - Survey of LLMs for code.
  
- [2023/11] **A Survey on Language Models for Code** [[paper](https://arxiv.org/pdf/2311.07989)]
  - A Survey on Language Models for Code.
  
- [2023/10] **StarCoder: may the source be with you!** [[paper](https://arxiv.org/pdf/2305.06161)]
  - IMPORTANT. A typical example of code LLM.

- [2023/07] **Efficient Training of Language Models to Fill in the Middle** [[paper](https://arxiv.org/pdf/2307.04964)] [[code](https://www.github.com/openai/human-eval-infilling)]
  - IMPORTANT. They use filling in the model to train the LLMs. Suitable for code LLMs since it employ the context information.

- [2023/07] **CodeGen2: Lessons for Training LLMs on Programming and Natural Languages** [[paper](https://arxiv.org/pdf/2305.02309)] [[code](https://github.com/salesforce/CodeGen)]
  - Poor paper. But they point out that Prefix-LM is useless.
  
- [2023/06] **WizardLM: Empowering Large Language Models to Follow Complex Instructions** [[paper](https://arxiv.org/pdf/2304.12244)] [[code](https://github.com/nlpxucan/WizardLM)]
  - A model for instruction evolution to create more complex and balanced dataset.
  
- [2023/02] **CodeGen: An Open Large Language Model for Code with Multi-turn Program Synthesis** [[paper](https://arxiv.org/pdf/2203.13474)] [[code](https://github.com/salesforce/CodeGen)]
  - Salesforce's codegen model. A large language model for code with multi-turn program synthesis.

## 3. LLM for RTL Coding

- [2024/02] **AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs** [[paper](https://arxiv.org/pdf/2402.00386)]
  - Use LLMs to build automated SVA generation pipeline.

- [2024/02] **RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution** [[paper](https://arxiv.org/pdf/2312.08617)] [[code](https://github.com/hkust-zhiyao/RTL-Coder )]
  - Introduce a new LLM training scheme based on code quality feedback.

- [2023/11] **RTLLM: An Open-Source Benchmark for Design RTL Generation with Large Language Model** [[paper](https://arxiv.org/pdf/2308.05345)] [[code](https://github.com/hkust-zhiyao/RTLLM)]
  - Open source benchmark for RTL LLM debuggers.
  
- [2023/10] **VerilogEval: Evaluating Large Language Models for Verilog Code Generation** [[paper](https://arxiv.org/pdf/2309.07544)]
  - Employ LLMs to generate comprehensive evaluation dataset. VerilogEval-machine and     VerilogEval-human.  
  
- [2023/07] **VeriGen: A Large Language Model for Verilog Code Generation** [[paper](https://arxiv.org/pdf/2307.04964)] [[code](https://github.com/shailja-thakur/VGen)]
  - IMPORTANT. First work fine-tunes LLMs for debugging.
