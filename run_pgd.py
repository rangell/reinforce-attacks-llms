import csv
import gc
import logging
import os

import numpy as np
import torch
import wandb

from baselines.reinforce.pgd_adapter import PGD
from baselines.reinforce.judge import judge_results
from baselines.reinforce.pgd_utils import STEP_LVL_
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
        project='pgd-reinforce',
        group=None,
        mode=None,
        name=None,
        dir=None,
        tags=None,
        metrics=None,
    )
    data = dict(
        behaviors_path='data/behavior_datasets/harmbench_behaviors_text_all.csv',
        indices=list(range(15))
    )
    evaluation = dict(
        judge_gen_lengths=(129, 256, 512),
        reward='sigmoid_yes_odds',
        out_dir='traces',
    )
    attack = dict(
        target_model=dict(
            model_name_or_path="google/gemma-2b-it",
            chat_template="gemma",
            dtype="bfloat16",
            num_gpus=1,
        ),
        targets_path='data/optimizer_targets/harmbench_targets_text.json',
        num_steps=5_000,
        use_prefix_cache=True,
        eval_steps=100,
        eval_best=False,
        reinforce_config=dict(
            strategy="reinforce",
            gen_max_len=128,
            gen_topk=256,
            gen_temperatures=[0, 0.7],
            gen_in_parallel_over_temps=False,
            bsln_temperature_complete=0,
            bsln_reward_clamp_min=0.5,
            bsln_reward_clamp_max=1.0,
            include_most_harmful=True,
            include_most_harmful_likelihood_thres=0.75,
            include_most_harmful_min_reward_thres=0.2,
            add_nogen_reward_baseline=0.1,
            token_reinforce_weighting='token',
            normalize_reinforce_loss=True,
            judge_with_control=False,
            judge_loss_config=dict(
                name="cais/HarmBench-Mistral-7b-val-cls",
                reward='sigmoid_yes_odds',
                padding_side="right",)
        ),
        token_position_weighting_kwargs=dict(
            name='linear',
            first_last_ratio=5.0),
        early_stop_key='target_reinforce_agg_reinforce_loss',
        target_weight=0,
        judge_weight=1.5,
        control_weight=0.007,
        control_next_weight=0.05,
        control_nonrepeat_weight=0.01,
    )


@experiment.capture(prefix='data')
def get_data(indices, behaviors):
    return [behaviors[index] for index in indices]


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

    pgd = PGD()
    model = pgd.model
    tokenizer = pgd.tokenizer
    template = pgd.template
    device = model.device
    data = get_data(behaviors=behaviors)

    # ========== attack ========== #
    test_cases, logs = pgd.generate_test_cases_chunk(data, verbose=False)

    # ========== log results ========== #
    torch.cuda.empty_cache()
    gc.collect()

    inputs = [template.format(instruction=test_case) for test_case in test_cases]
    with torch.inference_mode():
        _, completions, _ = check_refusal_completions(model, tokenizer, inputs=inputs)
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    behaviors = [d['Behavior'] for d in data]

    eval_test_case_scores, judge = judge_results(
        behaviors, completions, judge_model='cais/HarmBench-Mistral-7b-val-cls', device=device, return_answer=True)
    for log, behavior in zip(logs, behaviors):
        eval_reward, _ = judge_results(behavior, log[STEP_LVL_]['generated_string'], device=device, judge=judge)
        for k, v in eval_reward.items():
            log[STEP_LVL_][f'eval_reward_{k}'] = v.tolist()
    del judge

    logging.info(f'Final test cases: {test_cases}')
    logging.info(f'Completions: {completions}')
    eval_rewards = [f'{", ".join([f"{r:.2f}" for r in reward])} @ {length}'
                    for length, (reward, _) in eval_test_case_scores.items()]
    logging.info(f'Eval Rewards: {", ".join(eval_rewards)}')
    logging.info(
        f'Eval Success: {", ".join([f'{succ} @ {length}' for length, (_, succ) in eval_test_case_scores.items()])}')
    if wandb.run is not None:
        wandb.log({f'final_eval_reward_{k}': v.mean() for k, (v, _) in eval_test_case_scores.items()} |
                  {f'eval_success_{k}': sum([s.lower() == 'yes' for s in succs])
                   for k, (_, succs) in eval_test_case_scores.items()})

    test_case_scores, judge = judge_results(
        behaviors, completions, judge_model='cais/HarmBench-Llama-2-13b-cls', device=device, return_answer=True)
    for log, behavior in zip(logs, behaviors):
        test_reward, _ = judge_results(behavior, log[STEP_LVL_]['generated_string'], device=device, judge=judge)
        for k, v in test_reward.items():
            log[STEP_LVL_][f'test_reward_{k}'] = v.tolist()

    test_rewards = [f'{", ".join([f"{r:.2f}" for r in reward])} @ {length}'
                    for length, (reward, _) in test_case_scores.items()]
    logging.info(f'Rewards: {", ".join(test_rewards)}')
    logging.info(
        f'Success: {", ".join([f'{succ} @ {length}' for length, (_, succ) in test_case_scores.items()])}')
    if wandb.run is not None:
        wandb.log({f'final_reward_{k}': v.mean() for k, (v, _) in test_case_scores.items()} |
                  {f'success_{k}': sum([s.lower() == 'yes' for s in succs])
                   for k, (_, succs) in test_case_scores.items()})

    # ========== save results ========== #
    experiment_id = 'default' if _config['db_collection'] is None else _config['db_collection']
    run_id = 'default' if _config['overwrite'] is None else _config['overwrite']
    out_dir = os.path.join(_config["evaluation"]["out_dir"], experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"traces_{run_id}")

    traces = dict(logs=logs, test_cases=test_cases, eval_test_case_scores=eval_test_case_scores,
                  test_case_scores=test_case_scores)

    np.savez_compressed(file_path, **traces)
