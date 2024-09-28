# LLM for RTL Debugging

## Table of Content (ToC)

- [LLM for RTL Debugging](#llm-for-rtl-debugging)
  - [Table of Content (ToC)](#table-of-content-toc)
  - [1. LLM Foundations](#1-llm-foundations)
  - [2. LLM for Coding](#2-llm-for-coding)
  - [3. LLM for RTL Coding](#3-llm-for-rtl-coding)

## 1. LLM Foundations

- [2024/09] **VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool** [[paper](https://arxiv.org/abs/2408.08927)]
  - VERY IMPORTANT. Use graph-based planning to guide the LLMs to generate the code. Graph of thoughts for task planning. Tool feedback for code generation.

- [2024/07] **Converging Paradigms: The Synergy of Symbolic and Connectionist AI in LLM-Empowered Autonomous Agents** [[paper](https://arxiv.org/abs/2407.08516)] 
  - IMPORTANT. Future direction for assertion based code generation.

- [2024/06] **Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models** [[paper](https://arxiv.org/abs/2406.11736)] [[code](https://github.com/xufangzhi/ENVISIONS)]
  - A framework to let LLMs to interact with the environment and learn from the feedback. Use contrastive learning to improve the performance.

- [2024/06] **DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence** [[paper](https://arxiv.org/abs/2406.11931)]
  - IMPORTANT. In terms of code preference data, although the code compiler itself can already provide 0-1 feedback (whether the code pass all test cases or not), some code prompts may have a limited number of test cases, and do not provide full coverage, and hence directly using 0-1 feedback from the compiler may be noisy and sub-optimal.

- [2024/06] **Apple Intelligence Foundation Language Models** [[paper](https://arxiv.org/pdf/2407.21075)]
  - Apple's LLM. A small LLM for mobile devices.

- [2024/06] **Q-Sparse: All Large Language Models can be Sparsely-Activated** [[paper](https://arxiv.org/pdf/2407.10969)]
  - Select top-k in all linear layers to sparse network.

- [2024/06] **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning** [[paper](https://arxiv.org/abs/2310.06694)]
  - Prune the LLMs, making sure a fix ratio to structured pruning.

- [2024/06] **PathReasoner: Modeling Reasoning Path with Equivalent Extension for Logical Question Answering** [[paper](https://arxiv.org/abs/2405.19109)]
  - Add a transformer layer to model and diffuse the reasoning path.

- [2024/06] **Advancing LLM Reasoning Generalists with Preference Trees** [[paper](https://arxiv.org/abs/2406.11931)] [[code](https://github.com/OpenBMB/Eurus)]
  - IMPORTANT. Use preference tree to guide the LLMs through the reasoning process.

- [2024/04] **Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study** [[paper](https://arxiv.org/pdf/2404.10719)]
  - IMPORTANT. Key Factors to PPO for RLHF: (1) advantage normalization, (2) large- batch-size training, and (3) updating the parameters of the reference model with exponential moving average.

- [2024/03] **Extensive Self-Contrast Enables Feedback-Free Language Model Alignment [[paper](https://arxiv.org/pdf/2404.00604)]
  - IMPORTANT. Use failed sft result as dpo negatives.

- [2024/02] **Generative Representational Instruction Tuning** [[paper](https://arxiv.org/abs/2402.09906)] [[code](https://github.com/ContextualAI/gritlm)]
  - IMPORTANT. Pretrained generative model could have the ability for embedding task by simple fine-tuning without performance loss.

- [2024/02] **Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models** [[paper](https://arxiv.org/abs/2311.09278)]
  - Create a benchmark for symbol LLM tasks.

- [2024/02] **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** [[paper](https://arxiv.org/abs/2402.17764)]
  - IMPORTANT. Use (-1, 0, 1) quantization to compress all linear layers in LLMs. 

-[2024/01] **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models** [[paper](https://arxiv.org/abs/2401.06066)]
  - Design common expert for all tasks, and task-specific expert for each task. And group the task-specific expert into groups.

- [2024/01] **Mixtral of Experts** [[paper](https://arxiv.org/pdf/2401.04088)]  [[code](https://github.com/mistralai/mistral-src)]
  - One of the most popular open-source LLMs. A new model architecture that combines the strengths of transformer and mixture of experts.

- [2024/01] **Secrets of RLHF in Large Language Models Part II: Reward Modeling** [[paper](https://arxiv.org/pdf/2401.06080)]  [[code](https://github.com/OpenLMLab/MOSS-RLHF)]
  - IMPORTANT. Design superior reward dataset. (1) Label smothing; (2) Contrastive learning for improving; (3) Adaptive Margin; (4) Meta-learning for shifted reward distribution.

- [2023/12] **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** [[paper](https://openreview.net/pdf?id=5Xc1ecxO1h)]
    - IMPORTANT. Use tree structure to guide the LLMs to think and solve the problem.

- [2023/12] **DeepSeek: Towards Expert-Level Code Intelligence with Large Language Models** [[paper](https://arxiv.org/pdf/2312.08617)]

- [2023/08] **A Survey on Large Language Model based Autonomous Agents** [[paper](https://arxiv.org/abs/2308.11432)]

- [2023/08] **ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate** [[paper](https://arxiv.org/abs/2308.07201)]

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

- [2022/10] **Self-Instruct: Aligning Language Models with Self-Generated Instructions** [[paper](https://arxiv.org/pdf/2212.10560)] [[code](https://github.com/yizhongw/self-instruct)]
  - Boost the performace of LLM by the intruction generated by itself.

- [2022/10] **Emergent Abilities of Large Language Model** [[paper](https://arxiv.org/pdf/2206.07682)]
  - IMPORTANT. Large language models (LLMs) would acquire emergent abilities that are not explicitly trained for.

- [2022/09] **SGPT: GPT Sentence Embeddings for Semantic Search
** [[paper](https://arxiv.org/pdf/2202.08904)] [[code](https://github.com/Muennighoff/sgpt)]
  - Use generative GPT to act as a embedding model.

- [2022/05] **UL2: Unifying Language Learning Paradigms** [[paper](https://arxiv.org/pdf/2205.05131)] [[code](https://github.com/google-research/google-research/tree/master/ul2)]
  - Investigate and try to unify the pretraining paradigms.

- [2022/04] **PaLM: Scaling Language Modeling with Pathways** [[paper](https://arxiv.org/abs/2204.02311)] [[code](https://github.com/lucidrains/PaLM-pytorch)]
  - The bigest LLM I have seen.

- [2022/02] **UL2: Unifying Language Learning Paradigms.** [paper](https://arxiv.org/pdf/2205.05131) [[code](https://github.com/google-research/google-research/tree/master/ul2)]
  - They propose a new Mixture of Denoisers (MoD) pretraining that frames multiple pretraining
tasks as span corruption, diversifies and then mixes them.

- [2022/02] **Towards A Unified View of Parameter-Efficient Transfer Learning** [[paper](https://arxiv.org/pdf/2202.08390)] [[code](https://github.com/jxhe/unify-parameter-efficient-tuning)]
  - IMPORTANT. The adaptation framework for LLMs.

- [2022/02] **Finetuned Language Models are Zero-shot Learners** [[paper](https://arxiv.org/pdf/2109.01652)] [[code](https://github.com/google-research/flan)]
  - IMPORTANT. They propose Instruction Fine-tuning, which is a new fine-tuning strategy for LLMs. It is widely used in the field of LLMs.

- [2022/01] **Show Your Work: Scratchpads for Intermediate Computation with Language Models** [[paper](https://arxiv.org/abs/2112.00114)]
  - Add scratchpad to LLMs to store intermediate computation; something parallel with chain of thought.
    
- [2021/10] **LORA: Low-Rank Adaptation of Large Language Models** [[paper](https://arxiv.org/pdf/2106.09685)] [[code](https://github.com/microsoft/LoRA)]
  - An efficient method to fine-tune LLMs.

- [2021/03] **ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS** [[paper](https://arxiv.org/pdf/2003.10555)] [[code](https://github.com/google-research/electra)]
  - Use token discriminator to train BERT.

- [2020/01] **Scaling laws for neural language models** [[paper](https://arxiv.org/pdf/2001.08361)]
  - HARD and IMPORTANT. Investigate all kinds of power scaling of large language model based on transformer architecture.

- [2019/10] **Unified Language Model Pre-training for Natural Language Understanding and Generation** [[paper](https://arxiv.org/pdf/2005.14165)] [[code](https://github.com/microsoft/unilm)]
  - A mixed training strategy for both understand & generation task.

- [2019/10] **Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer** [[paper](https://arxiv.org/pdf/1910.10683)] [[code](https://github.com/google-research/text-to-text-transfer-transformer)]
  - IMPORTANT. Propose T5 which transform all NLP tasks into text-to-text format. The starter of prompt-based learning.

- [2019/02] **THE CURIOUS CASE OF NEURAL TEXT DeGENERATION** [[paper](https://arxiv.org/pdf/1904.09751)]
  - Repetitive and dull generation of LLMs. Propose nucleus sampling, which is widely used in the field of LLMs.
  
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

- [2022/11] **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning** [[paper](https://arxiv.org/pdf/2207.01780)] [[code](https://github.com/salesforce/CodeRL)]
  - IMPORTANT. Reinforcement learning with code LLMs. Use compiling and unit test signal to reward code generated model.

## 3. LLM for RTL Coding

- [2024/03] **HDLdebugger: Streamlining HDL debugging with Large Language Models** [[paper](https://arxiv.org/pdf/2403.11671)]  
  - Poor paper. Use many tricks to ext ract embedding. No public data  and code.

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
