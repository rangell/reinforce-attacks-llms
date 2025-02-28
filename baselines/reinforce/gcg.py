from collections import namedtuple
import gc
import itertools
import json
import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import wandb

from experiment import experiment
from ..baseline import SingleBehaviorRedTeamingMethod
from ..check_refusal_utils import check_refusal_completions
from .gcg_utils import interleave_lists, get_nonascii_toks, sample_control, x_bounded_sigmoid
from .generate_utils import generate, pad_and_concatenate
from .judge import Judge
from .prompt import PromptManager, get_template

torch.set_float32_matmul_precision("medium")
Generation = namedtuple('Generation',
                        ['id_', 'gen_ids', 'gen', 'reward', 'additional_rewards'],
                        defaults=[None])


# ============================== REINFORCE GCG CLASS DEFINITION ============================== #
class REINFORCE_GCG(SingleBehaviorRedTeamingMethod):

    @experiment.capture(prefix='attack')
    def __init__(self,
                 target_model: dict,
                 targets_path: str,
                 num_steps: int = 500,
                 limit_time_h: float | int | None = None,
                 warmup_num_steps: int = 0,
                 warmup_end_ce: float | None = None,
                 use_prefix_cache: bool = True,
                 adv_string_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 allow_non_ascii: bool = False,
                 filter_clean_up_tokenization_spaces: bool = False,
                 eval_steps: int = 10,
                 eval_best: bool = False,
                 search_width: int = 512,
                 search_n_replace: int = 1,
                 search_target_metric: str | None = 'agg_reinforce_loss',
                 search_monotonically: bool = False,
                 search_deterministic_loss: bool = True,
                 search_patience: int = -1,
                 search_greedy_reward_exploit_threshold: float = 0.8,
                 search_sample_strategy: str = 'gradient',
                 selection_extend_length_harmfulness: float = 0.01,
                 selection_starting_batch_size: int | None = 256,
                 early_stopping_abort: bool = False,
                 early_stopping_abort_min_loss: float = 0.1,
                 early_stopping_key: str | None = 'agg_reinforce_loss',
                 gen_max_len: int = 128,
                 gen_topk: int = 256,
                 gen_temperatures: list[int | float] = [0, 0.7],
                 gen_additional_judge_lengths: Sequence[int] = (20, 40, 80),
                 gen_in_parallel_over_temps: bool = False,
                 bsln_temperature_complete: int | float | None = 0,
                 bsln_reward_clamp_min: float = 0.5,
                 bsln_reward_clamp_max: float = 1.0,
                 bsln_ignore: bool = False,
                 include_most_harmful: bool = True,
                 include_most_harmful_min_reward_thres: float = 0.2,
                 include_most_harmful_likelihood_thres: float = 0.75,
                 add_nogen_reward_baseline: float = 0.1,
                 token_reinforce_weighting: str = 'token',
                 token_position_weighting_kwargs=dict(
                     name='linear',
                     first_last_ratio=5.
                 ),
                 normalize_reinforce_loss: bool = True,
                 judge_with_control: bool = False,
                 judge_weight: float = 1.0,
                 judge_loss_config=dict(
                     name="cais/HarmBench-Llama-2-13b-cls",
                     reward='sigmoid_yes_odds',
                     padding_side="right",
                 ),
                 target_weight: float = 0.0,
                 **model_kwargs):
        """Attack a model using the Greedy Coordinate Gradient (GCG) method using the REINFORCE objective.

        :param target_model (dict): a dictionary specifying the target model (kwargs to load_model_and_tokenizer).
        :param targets_path (str): the path to the targets JSON file.
        :param num_steps (int, optional): the number of optimization steps to use. Defaults to 500.
        :param limit_time_h (float | int | None, optional): limit attack duration (hours) if not None. Defaults to None.
        :param warmup_num_steps (int, optional): number of steps using affirmative objective. Defaults to 0.
        :param warmup_end_ce (float | None, optional): abort warmup if affirmative objective falls below threshold. Defaults to None.
        :param use_prefix_cache (bool, optional): whether to use KV cache for static prefix tokens (WARNING: Not all models support this).
        :param adv_string_init (str, optional): the initial adversarial string to start with. Defaults to "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !".
        :param allow_non_ascii (bool, optional): if true also tokens with non-ascii chars. Defaults to False.
        :param filter_clean_up_tokenization_spaces (bool, optional): tokenizer option for filtering tokenization spaces. Defaults to False.
        :param eval_steps (int, optional): how often to run eval. Defaults to 10.
        :param eval_best (bool, optional): if true, always evaluate best (according to early_stopping_key). Defaults to False.
        :param search_width (int, optional): the number of candidates to sample at each step. Defaults to 512.
        :param search_n_replace (int, optional): number of tokens to replace in each step. Defaults to 1.
        :param search_target_metric (str | None, optional): search metric to use for selecting best mutation/candidate (lower is better). If None, use loss. Defaults to 'agg_reinforce_loss'.
        :param search_monotonically (bool, optional): only accept mutations that are better than the previous one (w.r.t. search_target_metric). Defaults to False.
        :param search_deterministic_loss (bool, optional): if True, select best candidate by not considering random generations. Defaults to True.
        :param search_patience (int, optional): number of steps until which we rest to last best (w.r.t. early_stopping_key). Defaults to -1.
        :param search_greedy_reward_exploit_threshold (float, optional): if greedy reward exceeds the threshold, do not allow falling it under the threshold again. Defaults to 0.8.
        :param search_sample_strategy (str, optional): if 'gradient' use GCG's mutation strategy, if 'random' sample mutation uniformly. Defaults to 'gradient'.
        :param selection_extend_length_harmfulness (float, optional): harmfulness threshold for the gradual extension of the selection. Defaults to 0.01.
        :param selection_starting_batch_size (int | None, optional): the initial search_batch_size; will auto reduce later in case of OOM. Defaults to 256.
        :param early_stopping_abort (bool, optional): if true, an alternate form of early stopping, based solely on early_stopping_min_loss. Defaults to False.
        :param early_stopping_abort_min_loss (float, optional): the loss at which to stop optimization if early_stopping_abort is True. Defaults to 0.1.
        :param early_stopping_key (str | None, optional): metric key for early stopping. Defaults to 'agg_reinforce_loss'.
        :param gen_max_len (int, optional): the length to terminate generation. Defaults to 128.
        :param gen_topk (int, optional): topk tokens considered for generation. Defaults to 256.
        :param gen_temperatures (list[int  |  float], optional): temperatures used for generation. Exactly 0 results in greedy generation. Defaults to [0, 0.7].
        :param gen_additional_judge_lengths (Sequence[int], optional): additional lengths to obtain intermediate rewards. Only effective/tested with token_reinforce_weighting='token'. Defaults to (20, 40, 80).
        :param gen_in_parallel_over_temps (bool, optional): if True we will generate in parallel for multiple temperatures. May result in slightly different (greedy) generations to False. Defaults to False.
        :param bsln_temperature_complete (int | float | None, optional): temperature used to harmonize the length of bsln with gen_max_len. Defaults to 0.
        :param bsln_reward_clamp_min (float, optional): min reward the (non-completed) bsln generation will have. Defaults to 0.5.
        :param bsln_reward_clamp_max (float, optional): min reward the (non-completed) bsln generation will have. Defaults to 1.0.
        :param bsln_ignore (bool, optional): if True, the bsln will not be included. Defaults to False.
        :param include_most_harmful (bool, optional): if True, include the most likely harmful generation. Defaults to True.
        :param include_most_harmful_min_reward_thres (float, optional): min harmfulness to include most harmful generation. Defaults to 0.2
        :param include_most_harmful_likelihood_thres (float, optional): harmfulness beyond which only likelihood and lengths matter. Defaults to 0.75..
        :param add_nogen_reward_baseline (float, optional): if True, include a dummy generation with specified reward. Defaults to 0.1.
        :param token_reinforce_weighting (str, optional): if 'token', calculate reinforce weights for each token position where the max reward for the current state is used. If 'uniform', only use terminal rewards. Defaults to 'token'.
        :param token_position_weighting_kwargs (dict, optional): weight the tokens 'uniform'-ly or 'linear'-ly. Defaults to dict(name='linear', first_last_ratio=5.).
        :param normalize_reinforce_loss (bool, optional): if True, it is made sure that the absolute weights for each token position sum up to one. Defaults to True.
        :param judge_with_control (bool, optional): if True, we also include adversarial suffix in the prompt for the judge. If False, only the behavior is used. Defaults to False.
        :param judge_weight (float, optional): loss weight for the reinforce loss. Defaults to 1.0.
        :param judge_loss_config (dict, optional): configuration that is being passed to `Judge`. Defaults to dict( name="cais/HarmBench-Llama-2-13b-cls", reward='sigmoid_yes_odds', padding_side="right", ).
        :param target_weight (float, optional): loss weight for affirmative objective. Defaults to 0.0.
        """
        # assigns self.model, self.tokenizer, and self.model_name_or_path
        super().__init__(target_model=target_model, **model_kwargs)

        # ========== set attributes/configuration ========== #
        self.num_steps = num_steps
        self.limit_time_h = limit_time_h

        self.warmup_num_steps = warmup_num_steps
        self.warmup_end_ce = warmup_end_ce

        self.use_prefix_cache = use_prefix_cache
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            logging.warning(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache

        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii

        self.filter_clean_up_tokenization_spaces = filter_clean_up_tokenization_spaces

        self.eval_steps = eval_steps
        self.eval_best = eval_best

        self.search_width = search_width
        self.search_n_replace = search_n_replace
        self.search_target_metric = search_target_metric
        self.search_monotonically = search_monotonically
        self.search_deterministic_loss = search_deterministic_loss
        self.search_patience = search_patience
        self.search_greedy_reward_exploit_threshold = search_greedy_reward_exploit_threshold
        self.search_sample_strategy = search_sample_strategy

        self.selection_extend_length_harmfulness = selection_extend_length_harmfulness
        self.selection_starting_batch_size = selection_starting_batch_size

        self.early_stopping_abort = early_stopping_abort
        self.early_stopping_key = early_stopping_key
        self.early_stopping_abort_min_loss = early_stopping_abort_min_loss

        self.gen_max_len = gen_max_len
        self.gen_topk = gen_topk
        self.gen_temperatures = gen_temperatures
        self.gen_additional_judge_lengths = gen_additional_judge_lengths
        self.gen_in_parallel_over_temps = gen_in_parallel_over_temps

        self.bsln_temperature_complete = bsln_temperature_complete
        self.bsln_reward_clamp_min = bsln_reward_clamp_min
        self.bsln_reward_clamp_max = bsln_reward_clamp_max
        self.bsln_ignore = bsln_ignore

        self.include_most_harmful = include_most_harmful
        self.most_harmful_generation = None
        self.include_most_harmful_likelihood_thres = include_most_harmful_likelihood_thres
        self.include_most_harmful_min_reward_thres = include_most_harmful_min_reward_thres
        self.add_nogen_reward_baseline = add_nogen_reward_baseline

        self.token_reinforce_weighting = token_reinforce_weighting
        self.token_position_weighting_kwargs = token_position_weighting_kwargs

        self.normalize_reinforce_loss = normalize_reinforce_loss

        self.judge_with_control = judge_with_control
        self.judge_weight = judge_weight
        self.judge_loss_config = judge_loss_config

        self.target_weight = target_weight
        self.target_loss = CrossEntropyLoss(reduction='none')

        # ========== prepare attack resources ========== #
        self.model = self.model.eval()
        self.model.generation_config.cache_implementation = "static"  # Compile for generation
        logging.warning(f"WARNING: setting model.generation_config.cache_implementation=static")
        self.model.generation_config.compile_config.fullgraph = False  # Otherwise compile throws error for some models
        logging.warning(f"WARNING: setting model.generation_config.compile_config.fullgraph=False")

        # Instruction template
        template = get_template(**target_model)['prompt']
        self.template = template

        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)

        self.forward_loss_prefix = 'forward_'
        self.candidate_loss_prefix = 'candidate_'

        self.judge = Judge(**judge_loss_config, device=self.model.device, torch_dtype=self.model.dtype)

    def generate_test_cases_single_behavior(self, behavior_dict, verbose=True, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = (
            self.selection_starting_batch_size if self.selection_starting_batch_size else self.search_width)

        ### Targeted Model and Tokenizer ###
        device = self.model.device
        tokenizer = self.tokenizer
        embed_layer = self.model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(device))

        not_allowed_tokens = None if self.allow_non_ascii else get_nonascii_toks(tokenizer)
        not_allowed_tokens = torch.unique(not_allowed_tokens)  # remove redundant tokens

        # ========== Behavior and Target str ==========
        behavior = behavior_dict['Behavior']
        self.behaviors = [behavior]
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        target = self.behavior_id_to_target[behavior_dict["BehaviorID"]]
        self.target_strs = [target]

        behavior += ' '
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        # Reuse PGD implementation that ensures consistency between the string and token representation
        prompt_mgr = PromptManager(
            None, self.tokenizer, self.template, behavior_strs=self.behaviors, context_strs=[context_str],
            target_strs=self.target_strs, control_suffix_init=[self.adv_string_init], device=device)

        logging.info(f'Behavior: {behavior_id} || Target: {target}')

        # Resets
        self.bsln_generation = None  # reset at beginning of attack
        self.most_harmful_generation = None  # reset at beginning of attack
        self.default_gen_ids = None  # reset at beginning of attack

        # ========== setup optim_ids and cached components ========== #
        optim_ids = prompt_mgr.input_ids[prompt_mgr.c_suffix_batch_id, prompt_mgr.c_suffix_tok_id]
        num_optim_tokens = len(optim_ids[0])

        before_ids = prompt_mgr.input_ids[:, :prompt_mgr.c_suffix_tok_id[0, 0]]
        before_embeds = embed_layer(before_ids)
        after_ids = prompt_mgr.input_ids[:, prompt_mgr.c_suffix_tok_id[0, -1] + 1: prompt_mgr.target_tok_id[0, 0]]
        after_embeds = embed_layer(after_ids)
        target_ids = prompt_mgr.input_ids[:, prompt_mgr.target_tok_id[0, 0]:]

        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            with torch.no_grad():
                outputs = self.model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = [[tensor.clone() for tensor in layer] for layer in outputs.past_key_values]

        # ========== setup buffers ========== #
        all_losses = []
        all_target_metrics = []
        all_test_cases = []
        all_attacked = []
        all_completions = []
        all_steps = []
        all_runtimes = []
        all_log_dicts = []

        # ========== run optimization ========== #
        loss_dict = None
        loss = None
        optim_loss = float('inf')
        best_target_metric = float('inf')
        best_test_case = prompt_mgr.construct_instruction_str(prompt_mgr.hard_prompt()[0])[0]
        best_optim_ids = optim_ids
        best_step_or_reset = self.num_steps
        previous_optim_ids = optim_ids
        previous_greedy_reward = 0

        if self.limit_time_h is not None:
            attack_start_time = time.time()

        for i in tqdm(range(self.num_steps)):
            # To collect the runtime excluding evaluation
            start_step = torch.cuda.Event(enable_timing=True)
            start_step.record()

            if self.warmup_num_steps and i < self.warmup_num_steps:
                self.exploration_epoch = False
            else:
                self.exploration_epoch = True

            is_greedy_gen_inconsistent = False  # To check if generation from tokens and string is consistent
            # ========== compute coordinate token_gradient ========== #
            while True:  # to enable a fail over etc.
                optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=self.model.dtype)
                optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
                optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)
                input_ids = torch.cat([before_ids, optim_ids, after_ids], dim=1)
                if self.use_prefix_cache:
                    input_embeds_wo_target = torch.cat([optim_embeds, after_embeds], dim=1)
                else:
                    input_embeds_wo_target = torch.cat([before_embeds, optim_embeds, after_embeds], dim=1)

                loss, generations, gen_ids_embeds, forward_loss_dict = self.reinforce_forward(
                    step_idx=i, input_embeds_wo_target=input_embeds_wo_target,
                    input_ids=input_ids, target_ids=target_ids)
                token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

                # ========== logic that is conditional on current rewards/metrics ========== #
                greedy_pos = torch.where(torch.tensor([g.id_ == 0 for g in generations]))[0]
                greedy_reward = 0
                if len(greedy_pos) > 0:
                    greedy_reward = generations[greedy_pos].reward[0]
                    logging.info(f"\n===>Greedy (reward={greedy_reward:.3e}): {generations[greedy_pos].gen[0]}")
                    if ((((i - 1) % self.eval_steps == 0) or (i - 1 == self.num_steps - 1)) and i > 0
                            and not completions[0].startswith(generations[greedy_pos].gen[0])):
                        is_greedy_gen_inconsistent = is_greedy_gen_inconsistent is not None
                        if is_greedy_gen_inconsistent:  # only if it was not reset
                            logging.warning(f"\nInconsistency! Eval generation: {completions[0]} does not start with "
                                            f"forward generation {generations[greedy_pos].gen[0]}")

                if self.early_stopping_key and i >= self.warmup_num_steps:
                    current_metric = forward_loss_dict[f'{self.forward_loss_prefix}{self.early_stopping_key}'].item()
                    all_target_metrics.append(current_metric)
                    if current_metric < best_target_metric:
                        best_step_or_reset = i
                        best_target_metric = current_metric
                        best_optim_ids = optim_ids
                        best_test_case = prompt_mgr.construct_instruction_str(
                            prompt_mgr.hard_prompt(control_suffix_ids=best_optim_ids)[0])[0]
                    if self.early_stopping_abort and current_metric < self.early_stopping_abort_min_loss:
                        logging.info(f'Early stop attack at step {i} with {self.early_stopping_key}={current_metric}')
                        logging.info()
                        break
                else:
                    # Return latest if no early stopping key is provided
                    best_optim_ids = optim_ids
                    best_test_case = prompt_mgr.construct_instruction_str(
                        prompt_mgr.hard_prompt(control_suffix_ids=best_optim_ids)[0])[0]

                # Reset to last best if patience is reached
                if self.search_patience > 0 and i - best_step_or_reset >= self.search_patience:
                    logging.info('Out of patience; resetting to best')
                    optim_ids = best_optim_ids
                    best_step_or_reset = i
                    continue

                # Stop warm up if CE threshold is reached
                if (self.warmup_num_steps and i < self.warmup_num_steps and self.warmup_end_ce is not None and
                        forward_loss_dict[f'{self.forward_loss_prefix}ces'][0] < self.warmup_end_ce):
                    self.warmup_num_steps = 0

                # Switch to monotonic search if greedy response became harmful
                if (previous_greedy_reward > self.search_greedy_reward_exploit_threshold
                        and greedy_reward <= self.search_greedy_reward_exploit_threshold):
                    # reset to previous optim ids since greedy reward is below threshold
                    logging.info(f'Resetting to previous optim ids since new reward {greedy_reward} is too low. '
                                 f'Previous reward was {previous_greedy_reward}')
                    if (optim_ids != previous_optim_ids).any():
                        optim_ids = previous_optim_ids
                        is_greedy_gen_inconsistent = None  # No further comparison
                        continue
                    else:
                        logging.error('Recovering previous optim ids failed')

                # Break while if execution was successful and does not require a fail over
                break
            previous_greedy_reward = greedy_reward
            previous_optim_ids = optim_ids

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(
                optim_ids.squeeze(0), token_grad.squeeze(0), self.search_width, topk=256,
                n_replace=self.search_n_replace, temp=1, not_allowed_tokens=not_allowed_tokens,
                strategy=self.search_sample_strategy)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_cpu = sampled_top_indices.cpu()
            sampled_top_indices_text = tokenizer.batch_decode(
                sampled_top_indices_cpu, clean_up_tokenization_spaces=self.filter_clean_up_tokenization_spaces)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt",
                                add_special_tokens=False)['input_ids'][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices_cpu[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])

            if len(new_sampled_top_indices) == 0:
                logging.warning('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)

            if count >= self.search_width // 2:
                logging.info(f'Lots of removals: {count}')

            new_search_width = self.search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            with torch.inference_mode():
                if self.search_deterministic_loss:
                    # drop generations where the temperature is positive for a more stable/efficient candidate selection
                    # this will not update the reinforce weights and hence will be slightly incorrect
                    keep_mask = torch.tensor([not isinstance(g.id_, float) or g.id_ == 0 for g in generations])
                    generations = [g for m, g in zip(keep_mask, generations) if m]
                    gen_ids_embeds = gen_ids_embeds[keep_mask]
                    self.last_reinforce_weights = self.last_reinforce_weights[keep_mask]

                sampled_top_embeds = embed_layer(sampled_top_indices)
                # target embeds will be taken from gen ids embeds
                if self.use_prefix_cache:
                    input_embeds = torch.cat([sampled_top_embeds, after_embeds.repeat(new_search_width, 1, 1)], dim=1)
                else:
                    input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                              sampled_top_embeds,
                                              after_embeds.repeat(new_search_width, 1, 1)], dim=1)

                # Auto Find Batch Size for forward candidates (each time go OOM will decay search_batch_size // 2)
                loss, loss_dict = find_executable_batch_size(
                    self.reinforce_candidates_loss, self.search_batch_size)(input_embeds, generations, gen_ids_embeds)

                # ========== Update the optim_ids with the best candidate ========== #

                if self.search_target_metric and self.exploration_epoch:
                    torch.cuda.synchronize()  # avoid issues with the loss dict being loaded to cpu (non-blocking)
                    current_min_loss = loss_dict[
                        f'{self.candidate_loss_prefix}{self.search_target_metric}'].min().item()
                    current_argmin_loss = loss_dict[f'{self.candidate_loss_prefix}{self.search_target_metric}'].argmin()
                else:
                    current_min_loss = loss.min().item()
                    current_argmin_loss = loss.argmin()

                if self.search_monotonically:
                    if current_min_loss < optim_loss:
                        optim_loss = current_min_loss
                        optim_ids = sampled_top_indices[current_argmin_loss].unsqueeze(0)
                else:
                    optim_loss = current_min_loss
                    optim_ids = sampled_top_indices[current_argmin_loss].unsqueeze(0)

                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                runtime_step = start_step.elapsed_time(end)
                all_runtimes.append(runtime_step)
                logging.info(f'Step {i} took {runtime_step} ms')

                test_case = prompt_mgr.construct_instruction_str(
                    prompt_mgr.hard_prompt(control_suffix_ids=optim_ids)[0])[0]
                all_test_cases.append(test_case)

                all_losses.append(current_min_loss)

                # ========== Optionally generate ========== #
                test_dict = {}
                if (i % self.eval_steps == 0) or (i == self.num_steps - 1):
                    p_output = f'\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {optim_loss}'
                    input_str = self.template.format(instruction=best_test_case if self.eval_best else test_case)
                    is_refusal, completions, _ = check_refusal_completions(
                        self.model, tokenizer, inputs=[input_str])
                    all_completions.append(completions[0])
                    all_steps.append(i)

                    is_attacked = 0 if is_refusal[0] else 1
                    all_attacked.append(is_attacked)
                    p_output += f"\n\n===>Completion: {completions[0]}"
                    test_dict = {'loss': optim_loss, 'attacked': is_attacked, 'completion': completions[0]}

                    if verbose:
                        logging.info(p_output)

                # ========== Logging of step ========== #
                log_dict = test_dict | loss_dict | forward_loss_dict
                log_dict |= {'is_greedy_gen_inconsistent': is_greedy_gen_inconsistent}
                all_log_dicts.append(log_dict)

                if wandb.run is not None:
                    wandb.log({k: (wandb.Histogram(v.float()) if v.numel() > 1 else v.item())
                               if isinstance(v, torch.Tensor) else v
                               for k, v in log_dict.items()})

            if self.limit_time_h is not None and (time.time() - attack_start_time) / 3_600 > self.limit_time_h:
                logging.info(f'Limit time reached at step {i}')
                break

        logs = {'final_loss': optim_loss, 'all_losses': all_losses, 'all_test_cases': all_test_cases,
                'all_log_dicts': all_log_dicts, 'all_attacked': all_attacked, 'all_completions': all_completions,
                'all_steps': all_steps, 'all_runtimes': all_runtimes}
        return best_test_case, logs

    def reinforce_forward(
            self,
            step_idx: int,
            input_ids: torch.Tensor,
            input_embeds_wo_target: torch.Tensor,
            target_ids: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Generate responses
        with torch.inference_mode():
            # Handle bsln
            if self.default_gen_ids is None:
                gen_ids_shape = torch.Size([target_ids.shape[0], self.gen_max_len])
                self.default_gen_ids = torch.full(gen_ids_shape, self.tokenizer.eos_token_id, device=target_ids.device)
                self.default_gen_ids[:, :target_ids.shape[-1]] = target_ids

            generations = self.generations_with_reward(step_idx, input_ids, target_ids)

            # Weighted reward for leave-one-out reinforce-style loss
            reinforce_weights = self.calc_reinforce_weights(generations)
            if reinforce_weights.ndim == 3:
                reinforce_weights = reinforce_weights.squeeze(1)

        reinforce_weights = reinforce_weights.clone()  # clone since it is an inference tensor
        self.last_reinforce_weights = reinforce_weights

        target_ids_batch = torch.cat([generation.gen_ids.clone() for generation in generations])
        gen_ids_embeds = self.model.get_input_embeddings()(target_ids_batch)
        input_embeds_batch = torch.cat(
            [input_embeds_wo_target.repeat(len(generations), 1, 1), gen_ids_embeds], dim=1)

        tmp = input_embeds_batch.shape[1] - gen_ids_embeds.shape[1]
        logits = self.forward(input_embeds_batch)[..., tmp-1:-1, :].contiguous()

        ce_loss = self.target_loss(logits.transpose(1, 2), target_ids_batch)
        ce_weights = self.target_weights(target_ids_batch)
        ce_loss *= ce_weights
        reinforce_loss_per_gen = (ce_loss * reinforce_weights).sum(-1)
        reinforce_loss = reinforce_loss_per_gen.sum()
        assert self.target_weight == 0 or not self.bsln_ignore, "Baseline cannot be ignored with non-zero target weight"
        if self.exploration_epoch:
            loss_ = self.target_weight * ce_loss[0].sum() + self.judge_weight * reinforce_loss
        else:
            loss_ = ce_loss[0].sum()
        loss = loss_.sum()

        with torch.inference_mode():
            loss_dict = {}
            loss_dict['reinforce_loss'] = reinforce_loss
            loss_dict['reinforce_loss_per_gen'] = reinforce_loss_per_gen
            loss_dict["lens"] = torch.stack([(gen.gen_ids != self.tokenizer.eos_token_id).sum() for gen in generations])
            loss_dict["ces"] = ce_loss.sum(-1).detach()
            loss_dict["rewards"] = torch.cat([gen.reward for gen in generations])
            if self.gen_additional_judge_lengths is not None:
                additional_rewards = torch.stack([g.additional_rewards for g in generations])[:, 0]
                for l, r in zip(self.gen_additional_judge_lengths, additional_rewards.T):
                    loss_dict[f"additional_rewards_{l}"] = r
            self.aggregate_and_augment_loss_dict(loss_dict, generations)

        if self.include_most_harmful and self.exploration_epoch:
            self.remember_most_harmful_generation(generations, loss_dict)

        loss_dict_out = {f'{self.forward_loss_prefix}{k}': v.detach().to('cpu', non_blocking=True)
                         for k, v in loss_dict.items()}
        if len(generations[0].gen) == 1:  # TODO: check if the batch size will always be 1
            loss_dict_out |= {f'{self.forward_loss_prefix}{g.id_}_gen': g.gen[0] for g in generations}
            loss_dict_out |= {f'{self.forward_loss_prefix}{g.id_}_reward': g.reward[0].to('cpu', non_blocking=True)
                              for g in generations}

        return loss, generations, gen_ids_embeds, loss_dict_out

    def reinforce_candidates_loss(self, search_batch_size, input_embeds, generations, gen_ids_embeds):
        if self.search_batch_size != search_batch_size:
            logging.warning(f"Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        rewards = torch.cat([gen.reward for gen in generations])
        if self.gen_additional_judge_lengths is not None:
            additional_rewards = torch.stack([g.additional_rewards for g in generations])[:, 0]
        n_include = gen_ids_embeds.shape[1]
        # Check how long the sequence should for the selection of the best mutation
        if self.gen_additional_judge_lengths is not None and self.selection_extend_length_harmfulness > 0:
            # Exclude bsln and harm because they do not reflect the harmfulness of the model's current generations
            gen_id_include = torch.tensor([g.id_ not in ['bsln', 'harm'] for g in generations], device=rewards.device)
            is_term_above_threshold = (rewards[gen_id_include] > self.selection_extend_length_harmfulness).any()
            is_add_above_threshold = (additional_rewards[gen_id_include] >
                                      self.selection_extend_length_harmfulness).any(0)
            # Examples on how to truncate
            # is_add_above_threshold: False, False, True    -> max (unchanged)
            # is_add_above_threshold: True, False, True     -> max (unchanged)
            # is_add_above_threshold: True, True, False     -> gen_additional_judge_lengths[2]
            # is_add_above_threshold: False, True, False     -> gen_additional_judge_lengths[2]
            # is_add_above_threshold: True, False, False    -> gen_additional_judge_lengths[1]
            # is_add_above_threshold: False, False, False   -> gen_additional_judge_lengths[1]
            if not is_term_above_threshold and not is_add_above_threshold[-1]:
                if (~is_add_above_threshold).any():
                    n_include_i = 1
                else:
                    n_include_r = torch.argmax(torch.flip(is_add_above_threshold[:, None], (0, 1))[..., 0].float())
                    n_include_i = (len(self.gen_additional_judge_lengths) - n_include_r - 1).clamp(min=1)
                n_include = min(self.gen_additional_judge_lengths[n_include_i], n_include)
            logging.info(f"Truncating target to {n_include} tokens")

        all_loss = []
        num_gens = gen_ids_embeds.shape[0]
        loss_dict_all = dict()
        for i in range(0, input_embeds.shape[0], search_batch_size):
            input_embeds_batch = input_embeds[i:i+search_batch_size]
            # candidates may get filtered before here. and search_batch_size > len(input_embeds)
            embeds_bs = input_embeds_batch.shape[0]
            # repeat the the input embeds num_gens times. BS*num_gens
            input_embeds_batch = input_embeds_batch.repeat_interleave(num_gens, 0)
            # repeat gen emebds bs times. num_gens*BS
            gen_ids_embeds_ = gen_ids_embeds.repeat(embeds_bs, 1, 1)
            # Cat gen embeds with the input embeds to obtain (here with `num_gens=3` & `b=embeds_bs`)
            # ```
            # Tensor([
            #     adv_suffix_1 | gen_1,
            #     adv_suffix_1 | gen_2,
            #     adv_suffix_1 | gen_3,
            #     adv_suffix_2 | gen_1,
            #     ...
            #     adv_suffix_b | gen_2,
            #     suffix_b | gen_3
            # ])
            # ```
            input_embeds_batch = torch.cat([input_embeds_batch, gen_ids_embeds_], dim=1)

            # Repeat target/gen ids to obtain (here with `num_gens=3`)
            # ```
            # Tensor([
            #     gen_id_1,
            #     gen_id_2,
            #     gen_id_3,
            #     gen_id_1,
            #     ...
            #     gen_id_2,
            #     gen_id_3
            #     ])
            # ```
            # [
            target_ids_batch = torch.cat([g.gen_ids for g in generations], dim=0)

            last_reinforce_weights = self.last_reinforce_weights
            if n_include != target_ids_batch.shape[-1]:
                # Truncate inputs
                n_total = input_embeds_batch.shape[1] - target_ids_batch.shape[1] + n_include
                target_ids_batch = target_ids_batch[..., :n_include]
                input_embeds_batch = input_embeds_batch[..., :n_total, :]
                last_reinforce_weights = last_reinforce_weights[..., :n_include]

            target_ids_batch = target_ids_batch.repeat(embeds_bs, 1)

            if self.model._supports_num_logits_to_keep():  # to save some memory
                num_logits_to_keep = target_ids_batch.shape[1] + 1
                logits = self.forward(input_embeds_batch, num_logits_to_keep=num_logits_to_keep)
                logits = logits[..., :-1, :].contiguous()
            else:
                length_until_target = input_embeds_batch.shape[1] - target_ids_batch.shape[1]
                logits = self.forward(input_embeds_batch)[..., length_until_target-1:-1, :].contiguous()
            ce_loss = self.target_loss(logits.transpose(1, 2), target_ids_batch).to(torch.float32)
            ce_loss = ce_loss.reshape(-1, num_gens, ce_loss.shape[-1])
            ce_weights = self.target_weights(
                target_ids_batch.reshape(-1, num_gens, target_ids_batch.shape[-1]))
            ce_loss *= ce_weights

            # Collect furhter losses/metrics
            loss_dict_batch = dict()
            loss_dict_batch["ces"] = ce_loss.sum(-1).transpose(0, 1)

            # TODO: can be used to determine efficiently if greedy response changes. However, greedy responses are brittle and there is no use yet
            # if len(generations) > 1:
            #     assert generations[1].id_ == 0, "Initial greedy generation must be the generation with array index 1"
            #     loss_dict_batch["is_greedy_gen_unchanged"] = (
            #         (target_ids_batch.reshape(-1, num_gens, target_ids_batch.shape[-1])[:, 1] ==
            #             logits.reshape(-1, num_gens, *logits.shape[-2:])[:, 1].argmax(-1))
            #         | (ce_weights[:, 1] <= 0)).any(-1).float()

            reinforce_weights = last_reinforce_weights.repeat(embeds_bs, 1, 1)
            reinforce_loss_per_gen = (ce_loss * reinforce_weights).sum(-1)
            reinforce_loss = reinforce_loss_per_gen.sum(-1)

            loss_dict_batch["reinforce_loss"] = reinforce_loss
            loss_dict_batch["reinforce_loss_per_gen"] = reinforce_loss_per_gen.T
            loss_dict_batch["rewards"] = rewards[:, None].repeat(1, embeds_bs)
            self.aggregate_and_augment_loss_dict(loss_dict_batch, generations)

            assert self.target_weight == 0 or not self.bsln_ignore, \
                "Baseline cannot be ignored with non-zero target weight"
            if self.exploration_epoch:
                loss = self.target_weight * loss_dict_batch["ces"][0] + self.judge_weight * reinforce_loss
            else:
                loss = loss_dict_batch["ces"][0]

            # Reshape etc tensors in loss_dict
            for k, v in loss_dict_batch.items():
                val_in_all = loss_dict_all.get(k)
                if v.dim() == 0:
                    v = v.unsqueeze(0)
                if v.dim() == 2:
                    v = v.transpose(0, 1)
                if val_in_all is not None:
                    loss_dict_all[k].append(v)
                else:
                    loss_dict_all[k] = [v]

            all_loss.append(loss)

            del logits, ce_loss

        for k, v in loss_dict_all.items():
            loss_dict_all[k] = torch.cat(loss_dict_all[k], dim=0).detach()

        mean_dict = {f"{k}_mean": v.mean(dim=0) for k, v in loss_dict_all.items()}
        loss_dict_all.update(mean_dict)

        loss_dict_out = {f'{self.candidate_loss_prefix}{k}': v.detach().to('cpu', non_blocking=True)
                         for k, v in loss_dict_all.items()}
        loss_dict_out[f'{self.candidate_loss_prefix}n_include'] = n_include

        return torch.cat(all_loss, dim=0), loss_dict_out

    def gen_fn(self, *args, temperature: float | int | torch.Tensor = 0., target_length: int, **kwargs) -> torch.Tensor:
        """Generate with model and apply config to satisfy huggingface interface"""
        # Prepare configuration
        kwargs.setdefault('max_new_tokens', target_length)
        if isinstance(temperature, torch.Tensor) or temperature > 0:
            kwargs.setdefault('top_k', self.gen_topk)
            kwargs = dict(**kwargs, do_sample=True, temperature=temperature)
        else:
            kwargs.setdefault('top_p', None)
            kwargs.setdefault('top_k', None)
            kwargs.setdefault('temperature', None)
            kwargs = dict(**kwargs, do_sample=False)

        if isinstance(temperature, torch.Tensor):
            # Parallel generation with multiple temperatures
            return generate(self.model, *args, **kwargs)
        else:
            return self.model.generate(*args, **kwargs)

    def generate_responses(self, input_ids: torch.Tensor, input_mask: torch.Tensor,
                           temperatures: List[float | int], target_length: int, gen_offset: int = 0) -> Tuple[
            List[Tuple[str | int | float, torch.Tensor]], List[str]]:
        bs = input_ids.shape[0]

        if self.gen_in_parallel_over_temps and len(temperatures) > 1:
            # Parallelize generation for all temperatures; hugging face models
            # do not support this?!
            n_temps = len(temperatures)
            temperatures_ = torch.tensor(
                temperatures, device=input_ids.device, dtype=torch.float32).repeat_interleave(bs)
            input_ids_ = input_ids.repeat(n_temps, 1)
            input_mask_ = input_mask.repeat(n_temps, 1)
            gen_ids = self.gen_fn(input_ids=input_ids_, attention_mask=input_mask_, target_length=target_length,
                                  temperature=temperatures_)
        else:
            gen_ids = pad_and_concatenate([
                self.gen_fn(input_ids=input_ids, attention_mask=input_mask, target_length=target_length,
                            temperature=temperature)
                for temperature in temperatures])
        if len(gen_ids) > 0:
            input_length = input_ids.shape[-1] + gen_offset
            gen_ids_cpu = gen_ids[:, input_length:].cpu()
            gens = self.tokenizer.batch_decode(gen_ids_cpu, clean_up_tokenization_spaces=False)
            gens = [g.lstrip() for g in gens]

            gen_ids = [(temp, gen_ids[i * bs:(i + 1) * bs, input_length:]) for i, temp in enumerate(temperatures)]
            gens = [gens[i * bs:(i + 1) * bs] for i in range(len(temperatures))]

            if self.gen_additional_judge_lengths:
                for length in self.gen_additional_judge_lengths:
                    gens_ = self.tokenizer.batch_decode(gen_ids_cpu[:, :length], clean_up_tokenization_spaces=False)
                    gens_ = [g.lstrip() for g in gens_]
                    gens_ = [gens_[i * bs:(i + 1) * bs] for i in range(len(temperatures))]
                    gens.extend(gens_)
        else:
            gens = []
        if self.tokenizer.eos_token is not None:
            # Split after first eos token
            gens = [[g.split(self.tokenizer.eos_token)[0] for g in gen] for gen in gens]
            for i, gen_id in gen_ids:  # eos out all indices after the first eos
                eos_mask = (gen_id == self.tokenizer.eos_token_id).cumsum(-1) > 0
                gen_id.masked_fill_(eos_mask, self.tokenizer.eos_token_id)
        return gen_ids, gens

    def generations_with_reward(
            self,
            step_idx: int,
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
            input_mask: torch.Tensor | None = None) -> List[Tuple[Any, Generation]]:
        bs = input_ids.shape[0]

        gen_temperatures = self.gen_temperatures

        # Only generate if in exploration phase
        if self.exploration_epoch:
            if input_mask is None:
                input_mask = torch.ones_like(input_ids)
            gen_ids, gens = self.generate_responses(input_ids, input_mask, gen_temperatures, self.gen_max_len)
            if self.bsln_temperature_complete is not None:  # ignored by default
                assert not self.bsln_ignore, "Baseline must not be ignored for completion"
                assert bs == 1, "Batch size must be 1 for completing the baseline"
                assert self.gen_additional_judge_lengths, "Additional judge lengths must be provided for completion"
                gen_max_len = self.gen_max_len - target_ids.shape[-1]
                if (target_ids.shape[-1] not in self.gen_additional_judge_lengths
                        and target_ids.shape[-1] < self.gen_max_len):
                    # replace closest gen_additional_judge_lengths with bsln length
                    # overwriting here yields slightly wrong results in first iteration
                    closest_add = min(self.gen_additional_judge_lengths, key=lambda x: abs(x - target_ids.shape[-1]))
                    logging.info(f"Replacing additional judge length {closest_add} with length {target_ids.shape[-1]}")
                    self.gen_additional_judge_lengths = [
                        e if e != closest_add else target_ids.shape[-1] for e in self.gen_additional_judge_lengths]
                complete_temp = [self.bsln_temperature_complete]
                input_ids_ = torch.cat((input_ids, target_ids), dim=-1)
                input_mask_ = torch.cat((input_mask, torch.ones_like(target_ids)), dim=-1)
                complete_gen_ids, complete_gens = self.generate_responses(
                    input_ids_, input_mask_, complete_temp, gen_max_len, gen_offset=-target_ids.shape[-1])
                gen_ids.insert(0, ('bsln', complete_gen_ids[0][1]))
                gens = interleave_lists(complete_gens, gens)
        else:
            gen_ids, gens = [], []

        if (self.bsln_generation is None and (self.bsln_temperature_complete is None or not self.exploration_epoch)
                and not self.bsln_ignore):
            # Default target string
            gen_ids.insert(0, ('bsln', self.default_gen_ids))
            gens.insert(0, self.target_strs)

        # Behaviors for calculating reward
        if self.judge_with_control:
            behaviors = self.tokenizer.decode(input_ids)[0]
        else:
            behaviors = self.behaviors

        if len(gens) > 0:
            # Parallelize reward calculation for all temperatures
            rewards = self.judge(len(gens) * behaviors, [g for l in gens for g in l])
            rewards = [rewards[i * bs:(i + 1) * bs] for i in range(len(gens))]

            # House-keeping for intermediate rewards
            add_rewards = {}
            add_default = None
            if self.gen_additional_judge_lengths:
                add_default = torch.zeros(
                    bs, len(self.gen_additional_judge_lengths), dtype=rewards[0].dtype, device=rewards[0].device)
                rewards_ = rewards[len(gen_ids):]
                if self.bsln_temperature_complete is None:
                    ids = gen_temperatures
                else:
                    ids = [id_ for id_, _ in gen_ids]

                for (length, id_), reward in zip(itertools.product(self.gen_additional_judge_lengths, ids), rewards_):
                    if id_ not in add_rewards:
                        add_rewards[id_] = []
                    if id_ == 'bsln':
                        # Only clamp min reward if below bsln length
                        clamp_min = self.bsln_reward_clamp_min if length <= target_ids.shape[-1] else 0.
                        reward.clamp_(min=clamp_min, max=self.bsln_reward_clamp_max)
                    add_rewards[id_].append(reward)

            if self.bsln_generation is None and self.bsln_temperature_complete is None and not self.bsln_ignore:
                # Only clamp min reward if below bsln length
                clamp_min = self.bsln_reward_clamp_min if target_ids.shape[-1] < self.gen_max_len else 0.
                rewards[0].clamp_(min=clamp_min, max=self.bsln_reward_clamp_max)

            # Create list of generation objects
            generations = [
                Generation(id_, i, g, r, torch.stack(add_rewards[id_]).T if id_ in add_rewards else add_default)
                for (id_, i), g, r in zip(gen_ids, gens, rewards)]
        else:
            generations = []

        # Retrieve/remember generation for default target string
        if self.bsln_generation is not None:
            generations.insert(0, self.bsln_generation)
        elif self.bsln_temperature_complete is None and not self.bsln_ignore:
            self.bsln_generation = generations[0]

        if self.exploration_epoch and self.include_most_harmful and self.most_harmful_generation is not None:
            generations.append(self.most_harmful_generation)

        # Truncate by max length to save compute, avoid if compilation to to varying shapes
        eos_masks = [g.gen_ids == self.tokenizer.eos_token_id for g in generations]
        max_len = max([torch.argmax(eos_mask.float()) if eos_mask.any() else eos_mask.shape[-1]
                       for eos_mask in eos_masks])
        logging.info(f'Max length of generation: {max_len}')
        generations = [g._replace(gen_ids=g.gen_ids[:, :max_len + 1]) for g in generations]

        # Harmonize lengths if one is shorter than the others
        lens = [g.gen_ids.shape[-1] for g in generations]
        max_len = max(lens)
        if not all([l == max_len for l in lens]):
            generations = [g._replace(gen_ids=F.pad(g.gen_ids, (0, max_len - g.gen_ids.size(-1)),
                                      value=self.tokenizer.eos_token_id)) for g in generations]

        return generations

    def calc_reinforce_weights(self, generations: List[Generation]) -> torch.Tensor:
        # Weighted reward for leave-one-out reinforce-style loss
        n_res = len(generations)
        eps = torch.finfo(generations[0].reward.dtype).eps
        if n_res > 1:
            if self.token_reinforce_weighting == 'uniform' or self.default_gen_ids is None:
                # calculate mean
                accum_reward = sum([r.reward for r in generations])
                if self.add_nogen_reward_baseline is not None:
                    accum_reward += self.add_nogen_reward_baseline
                    n_res += 1
                # Example with three variables:
                # a - (a + b + c - a) / 2, s = a + b + c
                # a - (s - a) / 2
                # (3 * a - s) / 2
                reinforce_weights = torch.stack(  # Add eps to avoid zero weights if all rewards equal the mean
                    [(n_res * r.reward - accum_reward) / (n_res - 1) / n_res for r in generations]) + eps
            elif self.token_reinforce_weighting == 'token':  # Tree based min max strategy to get token level rewards
                all_ids = torch.stack(  # shape B x L x S
                    [self.default_gen_ids if r.gen_ids is None else r.gen_ids for r in generations], dim=-1)
                rewards = torch.stack([r.reward for r in generations], dim=-1)  # shape B x S
                rewards = rewards[:, None, :].broadcast_to(all_ids.shape)  # shape B x L x S

                if self.gen_additional_judge_lengths:
                    rewards = rewards.clone()
                    positions = torch.tensor(self.gen_additional_judge_lengths, device=rewards.device) - 1
                    positions = torch.where(
                        positions >= rewards.shape[1], torch.tensor(0, device=rewards.device), positions)
                    for gen_pos, generation in enumerate(generations):
                        if generation.additional_rewards is None:
                            continue

                        rewards[:, positions, gen_pos] = generation.additional_rewards

                    # Propagate best reward to previous tokens (i.e, cummax from right to left)
                    rewards = torch.flip(torch.cummax(torch.flip(rewards, [0, 2, 1]), dim=1).values, [0, 2, 1])

                # Get max reward for each token position where tokens match
                agg_rewards = torch.zeros(
                    (*all_ids.shape[:-1], len(self.tokenizer)), dtype=rewards.dtype, device=all_ids.device)
                agg_rewards.scatter_reduce_(-1, all_ids, rewards, reduce="amax")
                token_rewards = torch.gather(agg_rewards, -1, all_ids)
                # Get min reward align sequence (-> only use better rewards if prefixes match)
                token_rewards = torch.cummin(token_rewards, dim=1).values

                accum_reward = token_rewards.sum(-1, keepdim=True)
                if self.add_nogen_reward_baseline is not None:
                    accum_reward += self.add_nogen_reward_baseline
                    n_res += 1

                # Add eps to avoid zero weights if all rewards equal the mean
                reinforce_weights = (n_res * token_rewards - accum_reward) / (n_res - 1) / n_res + eps

                reinforce_weights = reinforce_weights.permute(2, 0, 1)
            else:
                raise NotImplementedError(f"Token weighing strategy {self.token_reinforce_weighting} not implemented")
        else:
            reinforce_weights = torch.stack([r.reward for r in generations])

        if self.normalize_reinforce_loss:
            accum_weights = reinforce_weights.abs().sum(0, keepdim=True)
            reinforce_weights /= accum_weights + eps

        return reinforce_weights

    @torch.no_grad()
    def remember_most_harmful_generation(
            self, generations: List[Generation], loss_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        assert not self.judge_with_control, 'Only supported for judge loss without control'

        ces = loss_dict['ces']
        lens = loss_dict['lens']
        mask = torch.tensor([g.id_ != 'bsln' for g in generations])
        generations = [g for m, g in zip(mask, generations) if m]
        ces = ces[mask]
        lens = lens[mask]
        rewards = torch.stack([r.reward for r in generations])

        if (self.include_most_harmful_min_reward_thres is not None
                and (rewards < self.include_most_harmful_min_reward_thres).all().cpu()):
            # Return if no harmful (enough) sequence was found
            return
        else:
            most_harmful_idx = rewards.argmax(0)

        # Avoid replacing previously most harmful with harmful response that is very unlikely
        if self.include_most_harmful_likelihood_thres < 1 and self.most_harmful_generation is not None:
            last_harmful_idx = [i for i, g in enumerate(generations) if g.id_ == 'harm'][0]
            above_threshold = self.most_harmful_generation.reward > self.include_most_harmful_likelihood_thres
            ce_better = ces[last_harmful_idx] > ces[most_harmful_idx]
            most_harmful_idx[above_threshold & ~ce_better] = last_harmful_idx

            if (most_harmful_idx == last_harmful_idx).all():
                return

        gen_ids = [(self.default_gen_ids[pos_idx]
                    if generations[harmful_idx].gen_ids is None
                    else generations[harmful_idx].gen_ids[pos_idx])
                   for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        gen_ids = torch.stack(gen_ids)
        gen = [generations[harmful_idx].gen[pos_idx] for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        reward = [generations[harmful_idx].reward[pos_idx] for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        additional_rewards = None
        if self.gen_additional_judge_lengths:
            additional_rewards = torch.stack([generations[harmful_idx].additional_rewards[pos_idx]
                                              for pos_idx, harmful_idx in enumerate(most_harmful_idx)])
        self.most_harmful_generation = Generation('harm', gen_ids, gen, torch.stack(reward), additional_rewards)

    def forward(self, input_embeds, **kwargs):
        if self.use_prefix_cache:
            # expand for batch size.
            current_batch_size = input_embeds.shape[0]
            prefix_cache_batch = []
            for i in range(len(self.prefix_cache)):
                prefix_cache_batch.append([])
                for j in range(len(self.prefix_cache[i])):
                    prefix_cache_batch[i].append(self.prefix_cache[i][j].clone().expand(current_batch_size, -1, -1, -1))
            outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch, **kwargs)
        else:
            outputs = self.model(inputs_embeds=input_embeds, **kwargs)
        return outputs.logits

    def aggregate_and_augment_loss_dict(
            self,
            loss_dict: Dict[str, torch.Tensor],
            generations: List[Generation]):
        # Break down losses for each generation
        loss_dict.update(
            {f'{k}_{generations[i].id_}': v[i, ...] for k, v in loss_dict.items()
             if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == len(generations)
             for i in range(v.shape[0])})

        # Filter non-deterministic scores
        ids = [g.id_ for g in generations if g.id_ == 0 or isinstance(g.id_, str)]
        ids_pos = [pos for pos, g in enumerate(generations) if g.id_ == 0 or isinstance(g.id_, str)]
        if not any([id_ == 0 for id_ in ids]):
            return
        # params
        factor_weights_greedy_not_harmful = 1
        factor_weights_greedy_harmful = 2
        offset_greedy_not_harmful = 10
        # assume that reward is in [0, 1]
        # if greedy generations is not harmful, we want to emphasize making the harmful generations more likely
        rewards = loss_dict['rewards'][ids_pos]
        greedy_pos = torch.where(torch.tensor([id_ == 0 for id_ in ids]))[0]
        greedy_reward = rewards[greedy_pos]
        weights_greedy_not_harmful = ((factor_weights_greedy_not_harmful - 1) * rewards + 1)
        # once the greedy generations is harmful, we want to emphasize making the greedy generation more likely
        weights_greedy_harmful = torch.ones_like(rewards)  # reward \in [0, 1]
        weights_greedy_harmful[greedy_pos] = factor_weights_greedy_harmful
        # for selecting the best generation, having more harmful generations should lower the value or not affect it
        weights_greedy_not_harmful /= weights_greedy_not_harmful.sum(0, keepdim=True)
        weights_greedy_harmful /= weights_greedy_harmful.sum(0, keepdim=True)
        reinforce_loss_weights = (greedy_reward * weights_greedy_harmful
                                  + (1 - greedy_reward) * weights_greedy_not_harmful).float()
        agg_reinforce_loss = (loss_dict['reinforce_loss_per_gen'][ids_pos] * reinforce_loss_weights).sum(0)
        # for selecting the best generation, the value should be lower if the greedy generation is harmful
        agg_reinforce_loss += (1 - greedy_reward.squeeze(0)) * offset_greedy_not_harmful

        loss_dict.update({
            'agg_reinforce_loss': agg_reinforce_loss,       # lower -> more harm
        })

    def target_weights(self, target_ids: torch.Tensor, exclude_eos_pos: int = 0):
        """Generate target mask for target ids. Except for bsln generation, we want to include the first eos token.

        Args:
            target_ids (torch.Tensor): target ids of shape [..., B, T].
            exclude_eos_pos (int, optional): Exclude eos token at this position for dim B. Defaults to 0.

        Returns:
            torch.Tensor: target weights of shape [..., B, T].
        """
        device = target_ids.device
        func_name = self.token_position_weighting_kwargs.get('name', 'uniform')

        target_mask = torch.ones_like(target_ids, dtype=torch.float32)
        eos_mask = (target_ids == self.tokenizer.eos_token_id).cumsum(-1) > 0
        first_eos_pos = eos_mask.float().argmax(-1)
        first_eos_pos[..., exclude_eos_pos] = 0
        eos_mask.scatter_(-1, first_eos_pos[..., None], False)
        target_mask[eos_mask] = False

        if func_name == 'uniform':
            target_mask = target_mask / target_mask.sum(-1, keepdim=True)
            return target_mask
        elif func_name == 'linear':
            intercept = self.token_position_weighting_kwargs['first_last_ratio']
            slope = (1 - intercept) / (self.gen_max_len - 1)
            target_weights = torch.arange(self.gen_max_len, device=device)[:target_mask.shape[-1]]
            target_weights = intercept + slope * target_weights
            target_weights = target_mask * target_weights[None]
            target_weights /= target_weights.sum(-1, keepdims=True)
            return target_weights
        else:
            raise ValueError(f"Unknown token weighting function {func_name}")
