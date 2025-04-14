import argparse
import itertools
import json
import os
import tempfile


SBATCH_TEMPLATE = """
#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:__num_gpus__
#SBATCH --mem=16G
#SBATCH --time=0-6:00:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K-reinforce.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c "source /ext3/env.sh; python run_gcg.py with overwrite=__behavior_idx__ data.index=__behavior_idx__ data.behaviors_path='data/behavior_datasets/strong_reject_behaviors.csv' attack.target_model.model_name_or_path='__model_name__' __use_prefix_cache__"\
"""

if __name__ == "__main__":

    models = [
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/gemma2-2b_from_llama3-8b/checkpoint-40/", "google/gemma-2-2b-it")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/gemma2-2b_from_llama3-8b/checkpoint-320/", "google/gemma-2-2b-it")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/gemma2-2b_from_qwen2.5-7b/checkpoint-40/", "google/gemma-2-2b-it")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/gemma2-2b_from_qwen2.5-7b/checkpoint-320/", "google/gemma-2-2b-it")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/llama3.2-3b_from_gemma-7b/checkpoint-40/", "meta-llama/Llama-3.2-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/llama3.2-3b_from_gemma-7b/checkpoint-320/", "meta-llama/Llama-3.2-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/llama3.2-3b_from_qwen2.5-7b/checkpoint-40/", "meta-llama/Llama-3.2-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/llama3.2-3b_from_qwen2.5-7b/checkpoint-320/", "meta-llama/Llama-3.2-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/qwen2.5-3b_from_gemma-7b/checkpoint-40/", "Qwen/Qwen2.5-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/qwen2.5-3b_from_gemma-7b/checkpoint-320/", "Qwen/Qwen2.5-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/qwen2.5-3b_from_llama3-8b/checkpoint-40/", "Qwen/Qwen2.5-3B-Instruct")',
        '("/scratch/jb9146/git/stanford_alpaca/outputs/distillations/qwen2.5-3b_from_llama3-8b/checkpoint-320/", "Qwen/Qwen2.5-3B-Instruct")',
    ]
    behavior_indices = list(range(313))

    output_dir = "/scratch/rca9780/jailbreak_analysis_data/reinforce-gcg-logs/"

    for model_name in models:
        target_model_id = "---".join(model_name.replace("(", "").replace(")", "").replace("\"", "").split(", ")).replace("/", "--")[2:]

        for behavior_idx in behavior_indices:
            num_gpus = "1"

            job_name = "{}".format(target_model_id)
            out_path = "{}/{}".format(output_dir, target_model_id)

            sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
            sbatch_str = sbatch_str.replace("__out_path__", out_path)
            sbatch_str = sbatch_str.replace("__output_dir__", output_dir)
            sbatch_str = sbatch_str.replace("__model_name__", model_name.replace("\"", "\\\""))
            sbatch_str = sbatch_str.replace("__num_gpus__", num_gpus)
            sbatch_str = sbatch_str.replace("__behavior_idx__", str(behavior_idx))
            sbatch_str = sbatch_str.replace("__use_prefix_cache__", "attack.use_prefix_cache=False" if "google/gemma-2" in model_name else "")

            print(f"cmd: {model_name}\n")
            with tempfile.NamedTemporaryFile() as f:
                f.write(bytes(sbatch_str.strip(), "utf-8"))
                f.seek(0)
                os.system(f"sbatch {f.name}")

            exit()
