<div align="center">

<img style="max-width: 100%; width: 500px;" src="assets/reinforce_loss_vs_affirmative.png" alt="REINFORCE vs. Affirmative Objective"/>

</div>

# REINFORCE Attacks on Large Language Models

This repository contains the code for the 2025 paper (arXiv)
> REINFORCE Adversarial Attacks on Large Language Models: <br />
An Adaptive, Distributional, and Semantic Objective

This code base extends HarmBench's. While we deviate from HarmBench in executing experiments, the attacks are fully compatible. We provide an adapted HarmBench readme below that reflects the important changes in our attacks.

**Links:** [arXiv](https://arxiv.org/pdf/2502.17254) - [Project Page](https://www.cs.cit.tum.de/daml/reinforce-attacks-llms/)

Moreover, this repository also provides the official implementation for Projected Gradient Descent (PGD) from the 2024 paper (Next Gen AI Safety workshop @ ICML)
> Attacking Large Language Models with Projected Gradient Descent

**Links:** [arXiv](https://arxiv.org/abs/2402.09154) - [Workshop Paper](https://openreview.net/pdf?id=5OIC0yZpyG)

## Table of Contents

<!-- markdown-toc start -->
- [REINFORCE Attacks on Large Language Models](#reinforce-attacks-on-large-language-models)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Experiments](#experiments)
  - [Implementation Notes](#implementation-notes)
  - [Repository Structure](#repository-structure)
  - [Citation](#citation)
  - [🗡️ HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal 🛡️](#️-harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal-️)
    - [🗡️ What is HarmBench 🛡️](#️-what-is-harmbench-️)
    - [🌐 Overview 🌐](#-overview-)
    - [☕ Quick Start ☕](#-quick-start-)
      - [⚙️ Installation](#️-installation)
      - [🛠️ Running The Evaluation Pipeline](#️-running-the-evaluation-pipeline)
        - [Using run\_pipeline.py](#using-run_pipelinepy)
        - [Running individual steps](#running-individual-steps)
    - [🤗 Classifiers](#-classifiers)
    - [⚓ Documentation ⚓](#-documentation-)
    - [Citation](#citation-1)
<!-- markdown-toc end -->

## Installation

Here, we follow the official installation instructions and add `seml`:

```bash
git clone https://github.com/sigeisler/reinforce-attacks-llms
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install seml==0.5.4
```

We used the following software package versions in our experiments (using `conda`):
```
pytorch                   2.5.1           py3.12_cuda12.4_cudnn9.1.0_0    pytorch
pytorch-cuda              12.4                 hc786d27_7    pytorch
pytorch-mutex             1.0                        cuda    pytorch
torchaudio                2.5.1               py312_cu124    pytorch
torchtriton               3.1.0                     py312    pytorch
torchvision               0.20.1              py312_cu124    pytorch
transformers              4.47.1                   pypi_0    pypi
```

## Experiments

To start an experiment with **our** `REINFORCE-GCG`, you may execute
```bash
python run_gcg.py
```
and potentially overwrite configurations like
```bash
python run_gcg.py with wandb.enable=False attack.num_steps=10
```

Analogously, for **our** `REINFORCE-PGD` run:
```bash
python run_pgd.py
```
or regular `PGD`:
```bash
python run_pgd.py with reinforce_config.strategy=None
```

The main results can be reproduced via the configurations provided in `./experiments`. We recommend following the official documentation of the used experiment management tool `seml`. `seml` is a lightweight wrapper around `sacred` that implements a `mongodb` observer and simplifies the submission to `slurm`. Alternatively, we provide a parser of our configs that creates commands to reproduce the results. For example, the commands to reproduce the experiments with `REINFORCE-GCG` (Table 1) you may run:

```bash
python run_commands_for_config.py --config-file experiments/gcg_reinforce.yaml --output-file ./commands.sh
```

Then, you will need to execute the list of experiments (one per line) in `./commands.sh`.

To reproduce the main numbers for the respective experiments, you may use the following configs:
- Table 1: `gcg_reinforce.yaml`
- Table 2: `pgd_reinforce.yaml`
- Table 3/9: `gcg_reinforce_alttargets.yaml`
- Figure 3: `gcg_reinforce_limit_time.yaml`,  `pgd_reinforce_time.yaml`, and  `pgd_reinforce_ce_time.yaml`
- Table 4: `gcg_reinforce_ablation.yaml`
- Figure 4: `gcg_reinforce_search_strategy.yaml`


## Implementation Notes

The implementations for our attacks `REINFORCE-GCG` and `REINFORCE-PGD` are in `baselines/reinforce`. Here, `gcg.py` and `pgd_reinforce.py` contain the main implementation.

The code for `GCG` was originally developed for a different code base and `LitGPT` models. Most of its complexity stems from the handling of multiple prompts that are of different lengths, which requires some housekeeping. We extend the original implementation of PGD `pgd_attack.py` in `pgd_reinforce.py`. In `pgd_adapter.py`, we provide an adapter for the attacks interface of HarmBench. Instead of `SingleBehaviorRedTeamingMethod`, which is usually the base class for attacks, we extend `RedTeamingMethod` to allow batching multiple prompts/behaviors.

For `REINFORCE-GCG`, we adapt the code of HarmBench's `GCG` implementation in `baselines/gcg`. The implementation of `GCG` cannot be extended as we did it with `PGD`. Thus, we duplicate and adapt the `GCG` implementation. Even though there is a strong commonality between our `REINFORCE` losses for `GCG` and `PGD`, we ultimately decided to keep the implementations separate. In `GCG`, we optimize over a single prompt and must support intermediate rewards. In contrast, in `PGD`, we optimize over multiple prompts and solely have terminal rewards.

## Repository Structure

- `assets/`: assets for markdown/documentation
- `baselines/`: attack implementations
  - `baselines/reinforce/`: *Our* `PGD`, `REINFORCE-GCG`, and `REINFORCE-GCG` implementation
- `configs`: HarmBench's configs
- `data`: HammBench's and our data
- `experiments`: *Our* experiment configurations (`seml`)
- `harmbench_docs`: HarmBench's documentation
- `multimodalmodels`: HarmBench's multimodal implementation
- `scripts`: HarmBench's scripts for their pipeline
- `run_commands_for_config.py`: *Our* utility script for converting the `seml` configs in `experiments` into a list of commands
- `run_gcg.py`: *Our* experiment file for `GCG` experiments (generate test cases and their evaluation)
- `run_pgd.py`: *Our* experiment file for `PGD` experiments (generate test cases and their evaluation)

## Citation

If you use our work(s) or find the code useful, please cite

```
@inproceedings{geisler2025_reinforce_attacks_llms,
  title = {REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective},
  author = {Geisler, Simon and Abdalla, M. H. I. and Cohen-Addad, Vincent and Gasteiger, Johannes and G\"unnemann, Stephan},
  booktitle={arXiv:2502.17254},
  year = {2025},
}
```
and
```
@inproceedings{geisler2024_attacking_llms_pgd,
  title = {REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective},
  author = {Geisler, Simon and Abdalla, M. H. I. and Gasteiger, Johannes and G\"unnemann, Stephan},
  booktitle={Next Gen AI Safety workshop at ICML},
  year = {2024},
}
```

<hr style="height:10px;border:none;color:#333;background-color:#333;margin-bottom:1.5cm;margin-top:1.5cm">

## 🗡️ HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal 🛡️

<p align="center">

<img src="assets/harmbench_splash.png" alt="HarmBench"/>

</p>

**A fast, scalable, and open-source framework for evaluating automated red teaming methods and LLM attacks/defenses**

### 🗡️ What is HarmBench 🛡️

Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess these methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify key considerations previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of $18$ red teaming methods and $33$ target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses.

### 🌐 Overview 🌐
<div align="center">

<img src="assets/eval_pipeline-1.png" alt="HarmBench Evaluation Pipeline"/>

</div>

There are two primary ways to use HarmBench: (1) evaluating red teaming methods against a set of LLMs, and (2) evaluating LLMs against a set of red teaming methods. These use cases are both supported by the same evaluation pipeline, illustrated above. The process for running this evaluation pipeline is described below in the Quick Start section and in the [evaluation pipeline docs](./harmbench_docs/evaluation_pipeline.md)

HarmBench has out-of-the-box support for transformers-compatible LLMs, numerous closed-source APIs, and several multimodal models. To add new red teaming methods and models that aren't compatible with transformers, it may help to learn more about how the codebase is structured. We provide a set of docs to help with more advanced usage.


### ☕ Quick Start ☕


#### ⚙️ Installation

See above

#### 🛠️ Running The Evaluation Pipeline
The HarmBench evaluation pipeline consists of three steps (with an optional merging step), shown in the figure above: (1) generating test cases, (1.5) merging test cases, (2) generating completions, (3) evaluating completions. We provide two ways of running the evaluation pipeline:
- A higher-level run_pipeline.py utility that automates parts of the pipeline
- Scripts for running individual steps of the pipeline manually

##### Using run_pipeline.py
The easiest way to run the evaluation pipeline is with `./scripts/run_pipeline.py`. Examples are given below.

**Our methods** are available as `REINFORCE_GCG` and `REINFORCE_PGD` as well as `PGD`. The `behavior_chunk_size` config likely needs to be adapted to the available GPU memory and attacked model.
```bash
# Run GCG against all compatible models using a SLURM cluster
python ./scripts/run_pipeline.py --methods REINFORCE_GCG --models all --step all --mode slurm

# Run all compatible attacks against Llama 2 7B Chat using a SLURM cluster
python ./scripts/run_pipeline.py --methods all --models llama2_7b --step all --mode slurm

# ---

# Generate test cases for a subset of methods and models using a SLURM cluster
python ./scripts/run_pipeline.py --methods PGD,REINFORCE_GCG,REINFORCE_PGD --models baichuan2_7b,mistral_7b,llama2_70b --step 1 --mode slurm

# Merge test cases for the above methods on a local machine
python ./scripts/run_pipeline.py --methods PGD,REINFORCE_GCG,REINFORCE_PGD --models baichuan2_7b,mistral_7b,llama2_70b --step 1.5 --mode local

# Generate and evaluate completions for the above methods and models using a SLURM cluster
python ./scripts/run_pipeline.py --methods PGD,REINFORCE_GCG,REINFORCE_PGD --models baichuan2_7b,mistral_7b,llama2_70b --step 2_and_3 --mode slurm
```
In the `slurm` mode, jobs are submitted with appropriate dependencies. In the `local` mode, steps of the pipeline are sequentially run on the current machine. In the `local_parallel` mode, Ray is used to parallelize jobs across GPUs in a single machine without needing to install Slurm. We recommend using the `slurm` mode (or submitting a PR for a different cluster manager) to enable parallelizing test case generation across multiple machines. For single machines with multiple GPUs, we recommend using `local_parallel` mode. Additional details on how to use `run_pipeline.py` are in the [evaluation pipeline docs](./harmbench_docs/evaluation_pipeline.md).

##### Running individual steps
For more control over the evaluation pipeline, you can manually run individual steps using `scripts/generate_test_cases.sh`, `scripts/merge_test_cases.sh`, `scripts/generate_completions.sh`, and `scripts/evaluate_completions.sh`. For more information on how to do this, see the [evaluation pipeline docs](./harmbench_docs/evaluation_pipeline.md).

### 🤗 Classifiers
HarmBench provide 3 classifier models:
  - 🤗 [cais/HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) for standard and contextual behaviors. Example to use can be found [here](https://github.com/centerforaisafety/HarmBench/blob/main/notebooks/run_classifier.ipynb)
  - 🤗 [cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors) for multimodal behaviors. Example to use can be found [here](https://github.com/centerforaisafety/HarmBench/blob/main/notebooks/run_classifier_multimodal.ipynb)
  - 🤗 [cais/HarmBench-Mistral-7b-val-cls](https://huggingface.co/cais/HarmBench-Mistral-7b-val-cls) is a validation classifier and support standard, contextual and multimodal behaviors.

### ⚓ Documentation ⚓

For more details documentation about each component in the framework
  - [**Codebase Structure**](./harmbench_docs/codebase_structure.md)
  - [**Method and Model Configs**](./harmbench_docs/configs.md)
  - [**Evaluation Pipeline**](./harmbench_docs/evaluation_pipeline.md)

### Citation

If you find HarmBench useful in your research, you may cite the paper:

```
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```