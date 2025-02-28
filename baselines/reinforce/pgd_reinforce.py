from collections import namedtuple
import re
from typing import Any, Dict, List, Tuple

import torch

from experiment import experiment
from .generate_utils import generate, pad_and_concatenate
from .judge import Judge
from .pgd_attack import PGDMultiPromptAttack
from .prompt import PromptManager
from .pgd_utils import append_or_allocate, STEP_LVL_

Generation = namedtuple('Generation',
                        ['id_', 'gen_ids', 'gen', 'reward'])

PATTERN_REINFORCE_VARIANTS = re.compile(r'^target_reinforce_([\.0-9a-zA-Z]*)$')


class ReinforcePGDMultiPromptAttack(PGDMultiPromptAttack):

    @experiment.capture(prefix='attack.reinforce_config')
    def __init__(self,
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
                     padding_side="right",),
                 **kwargs):
        """Attack a model using the Projected Gradient Descent (PGD) method using the REINFORCE objective.

        :param gen_max_len (int, optional): the length to terminate generation. Defaults to 128.
        :param gen_topk (int, optional): topk tokens considered for generation. Defaults to 256.
        :param gen_temperatures (list[int  |  float], optional): temperatures used for generation. Exactly 0 results in greedy generation. Defaults to [0, 0.7].
        :param gen_in_parallel_over_temps (bool, optional): if True we will generate in parallel for multiple temperatures. May result in slightly different (greedy) generations to False. Defaults to False.
        :param bsln_temperature_complete (int | float | None, optional): temperature used to harmonize the length of bsln with gen_max_len. Defaults to 0.
        :param bsln_reward_clamp_min (float, optional): min reward the (non-completed) bsln generation will have. Defaults to 0.5.
        :param bsln_reward_clamp_max (float, optional): min reward the (non-completed) bsln generation will have. Defaults to 1.0.
        :param include_most_harmful (bool, optional): if True, include the most likely harmful generation. Defaults to True.
        :param include_most_harmful_min_reward_thres (float, optional): min harmfulness to include most harmful generation. Defaults to 0.2
        :param include_most_harmful_likelihood_thres (float, optional): harmfulness beyond which only likelihood and lengths matter. Defaults to 0.75..
        :param add_nogen_reward_baseline (float, optional): if True, include a dummy generation with specified reward. Defaults to 0.1.
        :param token_reinforce_weighting (str, optional): if 'token', calculate reinforce weights for each token position where the max reward for the current state is used. If 'uniform', only use terminal rewards. Defaults to 'token'.
        :param normalize_reinforce_loss (bool, optional): if True, it is made sure that the absolute weights for each token position sum up to one. Defaults to True.
        :param judge_with_control (bool, optional): if True, we also include adversarial suffix in the prompt for the judge. If False, only the behavior is used. Defaults to False.
        :param judge_loss_config (dict, optional): configuration that is being passed to `Judge`. Defaults to dict( name="cais/HarmBench-Llama-2-13b-cls", reward='sigmoid_yes_odds', padding_side="right", ).
        """
        super().__init__(**kwargs)

        # Configuration of reinforce attack
        self.gen_topk = gen_topk
        self.gen_temperatures = gen_temperatures
        self.normalize_reinforce_loss = normalize_reinforce_loss
        self.bsln_temperature_complete = bsln_temperature_complete
        self.token_reinforce_weighting = token_reinforce_weighting
        self.include_most_harmful = include_most_harmful
        self.include_most_harmful_likelihood_thres = include_most_harmful_likelihood_thres
        self.include_most_harmful_min_reward_thres = include_most_harmful_min_reward_thres
        self.bsln_reward_clamp_min = bsln_reward_clamp_min
        self.bsln_reward_clamp_max = bsln_reward_clamp_max
        self.add_nogen_reward_baseline = add_nogen_reward_baseline
        self.gen_in_parallel_over_temps = gen_in_parallel_over_temps
        self.judge_with_control = judge_with_control
        self.target_length = self.reinforce_gen_max_len = gen_max_len
        self.judge_loss_config = judge_loss_config

        # Initialize judge
        self.reinforce_judge = Judge(**self.judge_loss_config, device=self.model.device)

        # Initialize variables
        self.reinforce_last_discrete_cat = None
        self.reinforce_last_generations = None
        self.reinforce_early_stopping_state = {}

    @torch.no_grad()
    def run(self,
            *args,
            **kwargs):
        self.most_harmful_generation = None  # reset at beginning of attack
        self.current_target_gen_ids = None  # reset at beginning of attack
        self.current_target_gen = None  # reset at beginning of attack
        self.reinforce_early_stopping_state = {}  # reset at beginning of attack
        self.default_gen_ids = None  # reset at beginning of attack

        return super().run(*args, **kwargs)

    @torch.enable_grad()
    def _grad_fn(self,
                 step_idx: int,
                 prompt_mgr: PromptManager,
                 embedding_factors: torch.Tensor,
                 embedding_mask: torch.Tensor | None,
                 log: dict = None,
                 verbose: int = 1) -> Dict[str, torch.Tensor]:
        """Override default behavior to implement the reinforce strategy."""
        if prompt_mgr.target_length > self.reinforce_gen_max_len:
            self.target_length = prompt_mgr.target_length
        else:
            self.target_length = self.reinforce_gen_max_len

        out = self.reinforce_forward(
            step_idx, prompt_mgr, embedding_factors, embedding_mask, log=log, verbose=verbose, logver_prefix='relaxed')

        return out[0]

    @torch.inference_mode()
    def discrete_loss(self,
                      step_idx: int,
                      prompt_mgr: PromptManager,
                      control_prefix_ids: torch.Tensor | None,
                      control_prefix_mask: torch.Tensor | None,
                      control_prefix_str: str | None,
                      control_suffix_ids: torch.Tensor | None,
                      control_suffix_mask: torch.Tensor | None,
                      control_suffix_str: str | None,
                      target_ids: torch.Tensor | None,
                      target_mask: torch.Tensor | None,
                      target_str: str | None,
                      log: dict = None) -> Tuple[Dict[str, float],
                                                 List[torch.Tensor]]:
        """Override default behavior to implement the reinforce strategy."""
        if prompt_mgr.target_length > self.reinforce_gen_max_len:
            self.target_length = prompt_mgr.target_length
        else:
            self.target_length = self.reinforce_gen_max_len

        discrete_cat = (
            control_prefix_ids, control_prefix_mask,
            control_suffix_ids, control_suffix_mask,
            target_ids, target_mask
        )

        loss_dict, bsln_ids, bsln_mask = self.reinforce_forward(
            step_idx, prompt_mgr, discrete_control_and_target=discrete_cat, log=log, logver_prefix='discrete')

        loss_dict = {
            k: v.detach().to('cpu', non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

        input_ids = [i[m.bool()].to('cpu', non_blocking=True) for i, m in zip(bsln_ids, bsln_mask)]
        # Remember discretization for next soft forward
        self.reinforce_last_discrete_cat = discrete_cat

        return loss_dict, input_ids

    def reinforce_forward(
            self,
            step_idx: int,
            prompt_mgr: PromptManager,
            embedding_factors: torch.Tensor | None = None,
            embedding_mask: torch.Tensor | None = None,
            discrete_control_and_target: Tuple[torch.Tensor | None, ...] | None = None,
            log: dict = None,
            verbose: int = 1,
            logver_prefix='target'
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Generate responses
        with torch.inference_mode():
            generations = None
            # Use "hard" prompt for generations since "soft" prompts for gen
            # are not supported by HF/LitGPT
            if discrete_control_and_target is None:
                if self.reinforce_last_discrete_cat is not None:
                    discrete_control_and_target = self.reinforce_last_discrete_cat
                    generations = self.reinforce_last_generations
                else:
                    # Only required in first iteration
                    discrete_control_and_target = self.simple_discretization(
                        prompt_mgr, embedding_factors, embedding_mask)

            # Generate generations (optionally if operating on relaxed prompt)
            if generations is None:
                discrete_input_ids, discrete_input_mask = prompt_mgr.hard_prompt(*discrete_control_and_target)

                # Calculate generations and rewards
                generations = self.generations_with_reward(
                    step_idx, prompt_mgr, discrete_input_ids, discrete_input_mask)

                self.reinforce_last_generations = generations

            # Remember padded bsln tokens
            if self.default_gen_ids is None:
                self.default_gen_ids = torch.full_like(generations[1].gen_ids, self.tokenizer.eos_token_id)
                self.default_gen_ids[:, :prompt_mgr.target_tok_id.shape[-1]] = prompt_mgr.input_ids[
                    prompt_mgr.target_batch_id, prompt_mgr.target_tok_id]

            # Weighted reward for leave-one-out reinforce-style loss
            reinforce_weights = self.calc_reinforce_weights(generations)

        # Clone to avoid inference tensors
        reinforce_weights = reinforce_weights.clone()
        generations = [g._replace(gen_ids=g.gen_ids.clone() if g.gen_ids is not None else None, reward=g.reward.clone())
                       for g in generations]
        if self.default_gen_ids.is_inference():
            self.default_gen_ids = self.default_gen_ids.clone()

        if embedding_factors is not None:
            if torch.is_grad_enabled():
                embedding_factors.requires_grad_()
                if embedding_mask is not None:
                    embedding_mask.requires_grad_()
        elif (step_idx + 1) % self.eval_steps == 0:
            greedy_pos = torch.where(torch.tensor([g.id_ == 0 for g in generations]))[0]
            greedy_reward = generations[greedy_pos].reward[0]
            self.print(f"\n===>Greedy[0] (reward={greedy_reward:.3e}): {generations[greedy_pos].gen[0]}")
            append_or_allocate(log[STEP_LVL_], 'forward_greedy_gens', generations[greedy_pos].gen)

        loss = 0
        loss_dict = {}
        for idx, generation in enumerate(generations):
            if self.bsln_temperature_complete is not None and generation.id_ == 'bsln':
                continue  # will be handled together with `bslnc`

            if embedding_factors is not None:  # Soft prompt
                embedding_factors_ = self.prepare_embedding_factors(prompt_mgr, embedding_factors)
                embedding_mask_ = self.prepare_embedding_mask(prompt_mgr, embedding_mask)
                control_and_target = self.split_control_and_target(prompt_mgr, embedding_factors_, embedding_mask_)
                model_input, targets = self.soft_model_input_and_target(
                    prompt_mgr, control_and_target, gen_ids=generation.gen_ids)
            else:  # Hard prompt
                control_and_target = discrete_control_and_target
                model_input, targets = self.hard_model_input_and_target(
                    prompt_mgr, control_and_target, gen_ids=generation.gen_ids)

            logits = self.forward_maybe_prefix_cache(prompt_mgr, model_input)

            loss_reinforce, loss_dict_reinforce = self.reinforce_loss(
                reinforce_weights[idx], logits, *targets, generation, 'target_reinforce')
            loss = loss + self.judge_weight * loss_reinforce
            loss_dict |= loss_dict_reinforce

            if isinstance(generation.id_, str) and generation.id_.startswith('bsln'):
                if generation.id_ == 'bslnc':
                    model_input, targets = self.hard_model_input_and_target(
                        prompt_mgr, discrete_control_and_target, gen_ids=None)

                    logits = logits[:, :model_input['input_ids'].shape[-1]]

                    assert generations[0].id_ == 'bsln', 'First generation must be bsln'
                    loss_reinforce, loss_dict_reinforce = self.reinforce_loss(
                        reinforce_weights[0], logits, *targets, generation._replace(id_='bsln'), 'target_reinforce')
                    loss = loss + self.judge_weight * loss_reinforce
                    loss_dict |= loss_dict_reinforce

                bsln_ids, bsln_mask = None, None
                if embedding_factors is None:
                    bsln_ids = model_input['input_ids']
                    bsln_mask = model_input['attention_mask']

                if embedding_factors is not None:
                    # Call regular loss (excl. reinforce loss)
                    loss_dict |= self.loss(logits, prompt_mgr, control_and_target, log=log, logver_prefix=logver_prefix)
                else:
                    control_and_target = (  # Only retain masks for loss
                        None, control_and_target[1], None, control_and_target[3], None, control_and_target[5])
                    # Call regular loss (excl. reinforce loss)
                    loss_dict |= self.loss(
                        logits,
                        prompt_mgr,
                        control_and_target,
                        model_input['input_ids'],
                        model_input['attention_mask'],
                        log=log,
                        logver_prefix=logver_prefix)

                loss += loss_dict['combined']

            if torch.is_grad_enabled() and embedding_factors is not None:
                loss_ = loss.sum()
                loss_.backward()
                loss = loss.detach()

        loss_dict['combined'] = loss
        self.aggregate_and_augment_loss_dict(loss_dict, generations)

        # Get most harmful generation and its reward
        if self.include_most_harmful and embedding_factors is None:
            out = self.remember_most_harmful_generation(prompt_mgr, control_and_target, generations, loss_dict)
            if out is not None:
                bsln_ids = out['input_ids']
                bsln_mask = out['attention_mask']

        # Log reinforce losses
        if log is not None:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and 'reinforce' in key:
                    append_or_allocate(log[STEP_LVL_], f'loss_{logver_prefix}_{key}', value)

        return loss_dict, bsln_ids, bsln_mask

    @torch.no_grad()
    def generations_with_reward(
            self,
            step_idx: int,
            prompt_mgr: PromptManager,
            input_ids: torch.Tensor,
            input_mask: torch.Tensor) -> List[Tuple[Any, Generation]]:
        bs = input_ids.shape[0]

        # Only generate if in exploration phase
        gen_ids, gens, self.input_strs = self.generate_responses(
            prompt_mgr, input_ids, input_mask, self.gen_temperatures, self.target_length, include_target=False)
        self.gens = gens[0]

        if self.bsln_temperature_complete is not None:
            target_mask = prompt_mgr.get_target_mask()
            target_length = self.target_length - target_mask.sum(-1).min()
            complete_temp = [self.bsln_temperature_complete]
            complete_gen_ids, complete_gens, _ = self.generate_responses(
                prompt_mgr, input_ids, input_mask, complete_temp, target_length, include_target=True)
            complete_gen_ids_raw = complete_gen_ids[0][1]
            # Stitch together complete generations
            complete_gen_mask = torch.ones((bs, self.target_length), dtype=bool, device=input_ids.device)
            complete_gen_mask[:, :target_mask.shape[-1]] = ~target_mask
            complete_gen_ids = torch.full_like(gen_ids[0][1], self.tokenizer.eos_token_id)
            complete_gen_ids[prompt_mgr.target_batch_id, torch.arange(prompt_mgr.target_length)[None]
                             ] = prompt_mgr.input_ids[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id]
            complete_gen_ids[complete_gen_mask] = complete_gen_ids_raw[
                torch.flip(complete_gen_mask, dims=(1,))[:, :target_length]]
            complete_gens = self.tokenizer.batch_decode(complete_gen_ids)
            gen_ids.append(('bslnc', complete_gen_ids))
            gens.append(complete_gens)

        # Default target string
        gen_ids.insert(0, ('bsln', None))
        gens.insert(0, prompt_mgr.target_strs)

        # Behaviors
        if self.judge_with_control:
            behaviors = prompt_mgr.construct_prompt_str(input_ids, input_mask)
        else:
            behaviors = prompt_mgr.behavior_strs

        # Parallelize reward calculation for all temperatures
        rewards = self.reinforce_judge(len(gens) * behaviors, [g for l in gens for g in l])
        rewards = [rewards[i * bs:(i + 1) * bs] for i in range(len(gens))]
        bsln_pos = torch.where(torch.tensor([id_ == 'bsln' for id_, _ in gen_ids]))[0]
        rewards[bsln_pos].clamp_(min=self.bsln_reward_clamp_min, max=self.bsln_reward_clamp_max)
        if self.bsln_temperature_complete is not None:
            bslnc_pos = torch.where(torch.tensor([id_ == 'bslnc' for id_, _ in gen_ids]))[0]
            rewards[bslnc_pos].clamp_(max=self.bsln_reward_clamp_max)

        merged = [Generation(id_, i, g, r) for (id_, i), g, r in zip(gen_ids, gens, rewards)]

        # Must be last for early stopping
        if self.include_most_harmful and self.most_harmful_generation is not None:
            merged.append(self.most_harmful_generation)

        return merged

    def reinforce_loss(self,
                       signed_weight: torch.Tensor,
                       logits: torch.Tensor,
                       gen_ids_pad: torch.Tensor,
                       gen_mask_pad: torch.Tensor,
                       gen_target_batch_id: torch.Tensor,
                       gen_target_tok_id: torch.Tensor,
                       generation: Generation,
                       loss_dict_prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits[gen_target_batch_id, gen_target_tok_id - 1]

        assert gen_mask_pad.any(1).all(), 'Every generation must have at least one token that is not masked out'

        if signed_weight.ndim == 1:
            signed_weight = signed_weight[:, None]
        if signed_weight.shape[1] > logits.shape[1]:
            signed_weight = signed_weight[:, :logits.shape[1]]

        # Calc cross entropies
        ce_loss = self.ce(logits.transpose(2, 1), gen_ids_pad)
        ce_loss = ce_loss * signed_weight * self.get_target_weights(gen_mask_pad)
        loss = ce_loss.sum(-1)

        id_ = generation.id_
        reward = generation.reward
        with torch.no_grad():
            ce_loss = (self.ce(logits.transpose(2, 1), gen_ids_pad) * gen_mask_pad).sum(-1) / gen_mask_pad.sum(-1)
            # Clamp to avoid infinity values
            length = gen_mask_pad.sum(-1)
            log_reward = reward.log().clamp(min=-1e10)
            log_em = -ce_loss * length
            em_prob = torch.exp(log_em)
            rem = reward * em_prob
            lrem = -log_reward - log_em

        loss_dict = {
            f'{loss_dict_prefix}_{id_}': loss,
            f'{loss_dict_prefix}_{id_}_ce': ce_loss,
            f'{loss_dict_prefix}_{id_}_len': length,
            f'{loss_dict_prefix}_{id_}_em_prob': em_prob,
            f'{loss_dict_prefix}_{id_}_reward': reward,  # higher -> more harm
            f'{loss_dict_prefix}_{id_}_rem': rem,  # higher -> more harm
            f'{loss_dict_prefix}_{id_}_lrem': lrem,  # lower -> more harm
        }

        return loss, loss_dict

    def default_bsln_target(self, prompt_mgr: PromptManager):
        gen_ids = prompt_mgr.input_ids[prompt_mgr.target_batch_id, prompt_mgr.target_tok_id]
        gen_mask_pad = prompt_mgr.get_target_mask()  # handles padding
        gen_target_batch_id = prompt_mgr.target_batch_id
        gen_target_tok_id = prompt_mgr.target_tok_id
        return gen_ids, gen_mask_pad, gen_target_batch_id, gen_target_tok_id

    def soft_model_input_and_target(
            self,
            prompt_mgr: PromptManager,
            control_and_target: Tuple[torch.Tensor | None, ...],
            gen_ids: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        if gen_ids is None:
            inputs_embeds, input_mask = prompt_mgr.soft_prompt(*control_and_target)
            gen_ids, gen_mask_pad, gen_target_batch_id, gen_target_tok_id = self.default_bsln_target(prompt_mgr)
        else:
            out = prompt_mgr.soft_prompt_replace_target(*control_and_target[:4], target_ids=gen_ids)
            (_, inputs_embeds, input_mask, gen_mask_pad, gen_target_batch_id,
             gen_target_tok_id) = out

        model_input = dict(inputs_embeds=inputs_embeds, attention_mask=input_mask)
        targets = gen_ids, gen_mask_pad, gen_target_batch_id, gen_target_tok_id

        return model_input, targets

    def hard_model_input_and_target(
            self,
            prompt_mgr: PromptManager,
            control_and_target: Tuple[torch.Tensor | None, ...],
            gen_ids: List[torch.Tensor] | None) -> Dict[str, torch.Tensor]:

        if gen_ids is None:
            input_ids, input_mask = prompt_mgr.hard_prompt(*control_and_target)
            gen_ids, gen_mask_pad, gen_target_batch_id, gen_target_tok_id = self.default_bsln_target(prompt_mgr)
        else:
            out = prompt_mgr.hard_prompt_replace_target(*control_and_target, target_ids=gen_ids)
            (input_ids, _, input_mask, gen_mask_pad, gen_target_batch_id, gen_target_tok_id) = out

        model_input = dict(input_ids=input_ids, attention_mask=input_mask)
        targets = gen_ids, gen_mask_pad, gen_target_batch_id, gen_target_tok_id

        return model_input, targets

    def simple_discretization(
            self, prompt_mgr: PromptManager, embedding_factors: torch.Tensor, embedding_mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor | None, ...]:

        embedding_factors_ = self.prepare_embedding_factors(prompt_mgr, embedding_factors)
        embedding_mask_ = self.prepare_embedding_mask(prompt_mgr, embedding_mask)
        control_and_target = self.split_control_and_target(prompt_mgr, embedding_factors_, embedding_mask_)

        discrete_control_and_target = [(f.argmax(-1) if f is not None else None, m > 0.5 if m is not None else None)
                                       for f, m in zip(control_and_target[0::2], control_and_target[1::2])]
        discrete_control_and_target = [e for t in discrete_control_and_target for e in t]
        return discrete_control_and_target

    def generate_responses(self, prompt_mgr: PromptManager, input_ids: torch.Tensor, input_mask: torch.Tensor,
                           temperatures: List[float | int], target_length: int, include_target: bool = False) -> Tuple[
                               List[Tuple[str | int | float, torch.Tensor]], List[str], List[str]]:
        bs = input_ids.shape[0]

        padding_side_orig = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        input_strs = prompt_mgr.construct_input_str(
            [i[m.bool()] for i, m in zip(input_ids, input_mask)], None, include_target=include_target)
        encoding = self.tokenizer(input_strs, padding=True, return_tensors='pt').to(self.model.device)
        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        self.tokenizer.padding_side = padding_side_orig

        if self.gen_in_parallel_over_temps and len(temperatures) > 1:
            # Parallelize generation for all temperatures; hugging face models
            # do not support this?!
            n_temps = len(temperatures)
            temperatures_ = torch.tensor(
                temperatures, device=input_ids.device, dtype=torch.float32).repeat_interleave(bs)
            input_ids_ = input_ids.repeat(n_temps, 1)
            attention_mask_ = attention_mask.repeat(n_temps, 1)
            gen_ids = self.gen_fn(input_ids=input_ids_, attention_mask=attention_mask_, target_length=target_length,
                                  temperature=temperatures_)
            gen_ids = gen_ids.clone()  # To avoid inference tensors
        else:
            gen_ids = pad_and_concatenate([
                self.gen_fn(input_ids=input_ids, attention_mask=attention_mask, target_length=target_length,
                            temperature=temperature)
                for temperature in temperatures], pad_to_length=target_length + input_ids.shape[-1])
        gens = self.tokenizer.batch_decode(gen_ids[:, input_ids.shape[-1]:], clean_up_tokenization_spaces=False)
        gens = [g.lstrip() for g in gens]

        gen_ids = [(temp, gen_ids[i * bs:(i + 1) * bs, input_ids.shape[-1]:])  # TODO: replace by torch.split?
                   for i, temp in enumerate(temperatures)]
        gens = [gens[i * bs:(i + 1) * bs] for i in range(len(temperatures))]
        if self.tokenizer.eos_token is not None:
            # Split after first eos token
            gens = [[g.split(self.tokenizer.eos_token)[0] for g in gen] for gen in gens]

        return gen_ids, gens, input_strs

    def gen_fn(self, *args, temperature: float | int | torch.Tensor = 0., target_length: int, **kwargs) -> torch.Tensor:
        """To satisfy huggingface interface"""
        kwargs.setdefault('max_new_tokens', target_length)
        if isinstance(temperature, torch.Tensor) or temperature > 0:
            kwargs.setdefault('top_k', self.gen_topk)
            kwargs = dict(**kwargs, do_sample=True, temperature=temperature)
            return generate(self.model, *args, **kwargs)
        else:
            kwargs.setdefault('top_p', None)
            kwargs.setdefault('top_k', None)
            kwargs.setdefault('temperature', None)
            kwargs = dict(**kwargs, do_sample=False)
            return self.model.generate(*args, **kwargs)

    def calc_reinforce_weights(self, generations: List[Generation]) -> torch.Tensor:
        # Weighted reward for leave-one-out reinforce-style loss
        # Example with three variables:
        # a - (a + b + c - a) / 2, s = a + b + c
        # a - (s - a) / 2
        # (3 * a - s) / 2
        n_res = len(generations)
        eps = torch.finfo(generations[0].reward.dtype).eps
        if n_res > 1:
            if self.token_reinforce_weighting == 'uniform':
                accum_reward = sum([r.reward for r in generations])
                if self.add_nogen_reward_baseline is not None:
                    accum_reward += self.add_nogen_reward_baseline
                    n_res += 1
                reinforce_weights = torch.stack(
                    [(n_res * r.reward - accum_reward) / (n_res - 1) / n_res for r in generations])
            else:  # Tree based min max strategy to get token level rewards
                all_ids = torch.stack(  # shape B x L x S
                    [self.default_gen_ids if r.gen_ids is None else r.gen_ids for r in generations], dim=-1)
                rewards = torch.stack([r.reward for r in generations], dim=-1)  # shape B x S
                rewards = rewards[:, None, :].broadcast_to(all_ids.shape)  # shape B x L x S

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

                reinforce_weights = (n_res * token_rewards - accum_reward) / (n_res - 1) / n_res
                reinforce_weights = reinforce_weights.permute(2, 0, 1)
        else:
            reinforce_weights = torch.stack([r.reward for r in generations])

        if self.normalize_reinforce_loss:
            accum_weights = reinforce_weights.abs().sum(0, keepdim=True)
            reinforce_weights /= accum_weights + self.eps

        return reinforce_weights

    @torch.no_grad()
    def remember_most_harmful_generation(
            self, prompt_mgr: PromptManager, control_and_target: Tuple[torch.Tensor | None, ...] | None,
            generations: List[Generation], loss_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        assert not self.judge_with_control, 'Only supported for judge loss without control'

        generations = [g for g in generations if g.id_ != 'bsln']

        rewards = torch.stack([r.reward for r in generations])
        most_harmful_idx = rewards.argmax(0)

        if self.include_most_harmful_min_reward_thres is not None and self.most_harmful_generation is None:
            assert generations[0].id_ == 0, 'Second generation must be greedy'
            # Set to greedy response if all generations are less harmful than threshold
            most_harmful_idx[(rewards < self.include_most_harmful_min_reward_thres).all(0)] = 0

        # Do not update if all generations are less harmful
        if (most_harmful_idx == 0).all():
            return

        ces = torch.stack([loss_dict[f'target_reinforce_{g.id_}_ce'] for g in generations])
        lens = torch.stack([loss_dict[f'target_reinforce_{g.id_}_len'] for g in generations])

        # Avoid replacing previously most harmful with harmful response that is very unlikely
        if self.include_most_harmful_likelihood_thres < 1 and self.most_harmful_generation is not None:
            last_harmful_idx = [i for i, g in enumerate(generations) if g.id_ == 'harm'][0]
            below_threshold = (rewards < self.include_most_harmful_min_reward_thres).all(0)
            above_threshold = self.most_harmful_generation.reward > self.include_most_harmful_likelihood_thres
            ce_better = ces[last_harmful_idx] > ces[most_harmful_idx, torch.arange(ces.shape[1])]
            most_harmful_idx[below_threshold | (above_threshold & ~ce_better)] = last_harmful_idx

        gen_ids = [(self.default_gen_ids[pos_idx]
                    if generations[harmful_idx].gen_ids is None
                    else generations[harmful_idx].gen_ids[pos_idx])
                   for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        gen_ids = torch.stack(gen_ids)
        gen = [generations[harmful_idx].gen[pos_idx] for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        reward = [generations[harmful_idx].reward[pos_idx] for pos_idx, harmful_idx in enumerate(most_harmful_idx)]
        self.most_harmful_generation = Generation('harm', gen_ids, gen, torch.stack(reward))

    @torch.no_grad()
    def aggregate_and_augment_loss_dict(
            self,
            loss_dict: Dict[str, torch.Tensor],
            generations: List[Generation],
            prefix: str = 'target_reinforce'):
        # To allow for early stopping on most harmful generation
        if self.include_most_harmful and generations[-1].id_ != 'harm' and 'harm' in self.esp.early_stop_key:
            early_stop_key = self.esp.early_stop_key.replace('harm', 'bsln')
            if early_stop_key in loss_dict:
                loss_dict[self.esp.early_stop_key] = loss_dict[early_stop_key]

        # Exclude random generations for a deterministic metric
        ids = [g.id_ for g in generations if g.id_ == 0 or isinstance(g.id_, str)]
        if len(ids) == 1:
            return

        # params
        factor_weights_greedy_not_harmful = 1
        factor_weights_greedy_harmful = 2
        offset_greedy_not_harmful = 10
        # assume that reward is in [0, 1]
        # if greedy generations is not harmful, we want to emphasize making the harmful generations more likely
        rewards = torch.stack([loss_dict[f'{prefix}_{id_}_reward'] for id_ in ids])
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
        agg_reinforce_loss = (
            torch.stack([loss_dict[f'{prefix}_{id_}'] for id_ in ids]) * reinforce_loss_weights).sum(0)
        # for selecting the best generation, the value should be lower if the greedy generation is harmful
        agg_reinforce_loss += (1 - greedy_reward.squeeze(0)) * offset_greedy_not_harmful

        loss_dict.update({
            f'{prefix}_agg_reinforce_loss': agg_reinforce_loss,  # lower -> more harm
        })

    def _replace_target(self, prompt_mgr: PromptManager, gen_ids: torch.Tensor, gen: List[str]):
        """Replace target with most harmful generation."""
        self.current_target_gen_ids = gen_ids
        self.current_target_gen = gen
        self.current_prompt_mgr = prompt_mgr
        prompt_mgr.replace_target(gen_ids, gen)

    def _where_generations(
            self, mask: torch.Tensor, true_option: List[Generation], false_option: List[Generation]) -> Generation:
        if true_option.gen_ids is not None or false_option.gen_ids is not None:
            true_option_gen_ids = true_option.gen_ids if true_option.gen_ids is not None else self.default_gen_ids
            false_option_gen_ids = false_option.gen_ids if false_option.gen_ids is not None else self.default_gen_ids
            gen_ids = torch.where(mask[:, None], true_option_gen_ids, false_option_gen_ids)
        else:
            gen_ids = None
        gen = [
            true_option.gen[pos_idx] if mask[pos_idx] else false_option.gen[pos_idx] for pos_idx, _ in enumerate(mask)]
        reward = torch.where(mask, true_option.reward, false_option.reward)
        return Generation(true_option.id_, gen_ids, gen, reward)
