import csv
import gc
import logging
import os

import numpy as np
import torch
import wandb

from baselines.reinforce.gcg import REINFORCE_GCG
from baselines.reinforce.judge import judge_results
from baselines.check_refusal_utils import check_refusal_completions
from experiment import experiment, wandb_initialize, manual_seed


# Configure the logging format
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)


@experiment.config
def default_config():
    seed = 42
    wandb = dict(
        enable=True,
        id=None,
        entity=None,
        project='gcg-reinforce',
        group=None,
        mode=None,
        name=None,
        dir=None,
        tags=None,
        metrics=None,
    )
    data = dict(
        behaviors_path='data/behavior_datasets/harmbench_behaviors_text_all.csv',
        index=0
    )
    evaluation = dict(
        judge_model='cais/HarmBench-Llama-2-13b-cls',
        # We use 129 since the length is also truncated w.r.t. the judge's bos token
        judge_gen_lengths=(129, 256, 512),
        reward='sigmoid_yes_odds',
        out_dir='traces',
    )
    attack = dict(
        target_model=dict(
            model_name_or_path="google/gemma-1.1-2b-it",
            chat_template="gemma",
            dtype="bfloat16",
            num_gpus=1,
        ),
        targets_path='./data/optimizer_targets/harmbench_targets_text.json',
        num_steps=500,
        limit_time_h=None,
        warmup_num_steps=0,
        warmup_end_ce=None,
        use_prefix_cache=True,
        adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        allow_non_ascii=False,
        filter_clean_up_tokenization_spaces=False,
        eval_steps=10,
        eval_best=False,
        search_width=512,
        search_n_replace=1,
        search_target_metric='agg_reinforce_loss',
        search_deterministic_loss=True,
        search_patience=-1,
        search_greedy_reward_exploit_threshold=0.8,
        search_sample_strategy='gradient',
        selection_extend_length_harmfulness=0.01,
        selection_starting_batch_size=256,
        early_stopping_abort=False,
        early_stopping_abort_min_loss=0.1,
        early_stopping_key='agg_reinforce_loss',
        gen_topk=256,
        gen_max_len=128,
        gen_temperatures=[0, 0.7],
        gen_additional_judge_lengths=(20, 40, 80),
        gen_in_parallel_over_temps=False,
        bsln_temperature_complete=0,
        bsln_reward_clamp_min=0.5,
        bsln_reward_clamp_max=1.0,
        bsln_ignore=False,
        include_most_harmful=True,
        include_most_harmful_likelihood_thres=0.75,
        include_most_harmful_min_reward_thres=0.2,
        replace_bsln_with_harmful=False,
        add_nogen_reward_baseline=0.1,
        token_reinforce_weighting='token',
        token_position_weighting_kwargs=dict(
            name='linear',
            first_last_ratio=5.
        ),
        normalize_reinforce_loss=True,
        judge_with_control=False,
        judge_weight=1.0,
        judge_loss_config=dict(
            name='cais/HarmBench-Llama-2-13b-cls',  # or "cais/HarmBench-Mistral-7b-val-cls"
            reward='sigmoid_yes_odds',
            padding_side="right",
        ),
        target_weight=0.0,
    )


@experiment.capture(prefix='data')
def get_behavior_dict(index, behaviors):
    return behaviors[index] | {"index": index}


@experiment.automain
def run(_config):
    logging.info(_config)

    # ========== init ========== #
    wandb_initialize(_config)
    with open(_config["data"]["behaviors_path"], 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)

    # Set random seed to becomes as reproducible as possible
    manual_seed()

    gcg = REINFORCE_GCG()
    model = gcg.model
    tokenizer = gcg.tokenizer
    template = gcg.template
    device = model.device
    data = get_behavior_dict(behaviors=behaviors)

    # ========== attack ========== #
    test_case, logs = gcg.generate_test_cases_single_behavior(data, verbose=True)

    # ========== log results ========== #
    torch.cuda.empty_cache()
    gc.collect()

    input_str = template.format(instruction=test_case)
    with torch.inference_mode():
        _, completion, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    test_case_score, judge = judge_results(data['Behavior'], completion, device=device, return_answer=True)

    logging.info(f'Final test case: {test_case}')
    logging.info(f'Completion: {completion[0]}')
    logging.info(
        f'Rewards: {", ".join([f'{reward.item(): .3e} @ {length}' for length, (reward, _) in test_case_score.items()])}')
    logging.info(
        f'Success: {", ".join([f'{succ[0]} @ {length}' for length, (_, succ) in test_case_score.items()])}')
    if wandb.run is not None:
        wandb.log({f'final_reward_{k}': v.item() for k, (v, _) in test_case_score.items()} |
                  {f'success_{k}': float(s[0].lower() == 'yes') for k, (_, s) in test_case_score.items()})

    all_completions = logs['all_completions']
    results, _ = judge_results(data['Behavior'], all_completions, device=device, judge=judge)

    if wandb.run is not None:
        x_name = 'attack_iteration'
        y_name = 'reward'
        for k, ys in results.items():
            data = [[x, y] for (x, y) in zip(logs['all_steps'], ys)]
            table = wandb.Table(data=data, columns=[x_name, y_name])
            wandb.log({
                f'test_judge_{k}': wandb.plot.line(table, x_name, y_name, title=f'test_judge_{k}')
            })

    # ========== save results ========== #
    experiment_id = 'default' if _config['db_collection'] is None else _config['db_collection']
    run_id = 'default' if _config['overwrite'] is None else _config['overwrite']
    out_dir = os.path.join(_config["evaluation"]["out_dir"], experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"traces_{run_id}")

    traces = dict(logs=logs, test_case=test_case, test_case_score=test_case_score, results=results)

    np.savez_compressed(file_path, **traces)
