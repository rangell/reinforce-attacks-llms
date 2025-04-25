import copy
from functools import partial
import json
import logging
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from experiment import experiment
from ..baseline import RedTeamingMethod
from ..model_utils import get_template, load_model_and_tokenizer
from .pgd_attack import PGDMultiPromptAttack
from .pgd_reinforce import ReinforcePGDMultiPromptAttack


# =============== PGD CLASS DEFINITION ============================== #
class PGD(RedTeamingMethod):
    """
    PGD wrapper for parallel attacks multiple behaviors, which implements both the affirmative and REINFORCE objective
    """

    @experiment.capture(prefix='attack')
    def __init__(self, target_model, targets_path, num_steps=5_000, eval_steps=250, num_test_cases_per_behavior=1,
                 test_cases_batch_size=50, reinforce_config=dict(strategy=None), **kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: number of attack iterations, defaults to 5_000
        :param eval_steps: how often to run eval, defaults to 250
        :param num_test_cases_per_behavior: the number of test cases to generate in a call to self.generate_test_cases, defaults to 1
        :param test_cases_batch_size: number of test cases to put in each batch, defaults to 50
        :param reinforce_config: configuration specific to REINFORCE loss. Use affirmative objective if reinforce_config.strategy=='None', use REINFORCE loss if reinforce_config.strategy=='reinforce'. Defaults to dict(strategy='None')
        """
        # Forward logs to stdout
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.info(f'PGD received kwargs {kwargs}')
        logging.info(f'PGD received reinforce_config {reinforce_config}')

        self.reinforce_config = reinforce_config
        self.pgd_config = kwargs

        model_kwargs = target_model
        model, tokenizer = load_model_and_tokenizer(**model_kwargs)
        self.model_name_or_path = model_kwargs['model_name_or_path']
        self.model = model
        self.tokenizer = tokenizer
        self.num_test_cases_per_behavior = num_test_cases_per_behavior
        self.test_cases_batch_size = test_cases_batch_size

        self.template = get_template(**target_model)['prompt']

        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)

        if reinforce_config['strategy'] == 'reinforce':
            PGDImpl = partial(ReinforcePGDMultiPromptAttack, **reinforce_config, **self.pgd_config)
        else:
            PGDImpl = PGDMultiPromptAttack
        self.pgd = PGDImpl(model=self.model, template=self.template, tokenizer=self.tokenizer,
                           num_steps=num_steps, eval_steps=eval_steps, **self.pgd_config)

        logging.info({k: v for k, v in self.pgd.__dict__.items() if k not in ['tokenizer', 'model']})

    def generate_test_cases(self, behaviors, verbose=False):
        """
        Generates test cases for the provided behaviors. The outputs of this method are passed to the
        save_test_cases method, which saves test cases and logs to disk.

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to print progress
        :return: a dictionary of test cases, where the keys are the behavior IDs and the values are lists of test cases
        """
        # set self.test_cases_batch_size and self.num_test_cases_per_behavior to 1 if not defined yet
        # (enables subclassing with a new __init__ function that doesn't define these attributes)
        if not hasattr(self, 'test_cases_batch_size'):
            self.test_cases_batch_size = 1
        if not hasattr(self, 'num_test_cases_per_behavior'):
            self.num_test_cases_per_behavior = 1

        repetitions = self.num_test_cases_per_behavior
        behaviors = [b for b in behaviors for _ in range(repetitions)]

        # break it down into batches
        bs = self.test_cases_batch_size
        behavior_chunks = split_into_chunks(behaviors, bs)

        all_test_cases = {}
        all_logs = {}
        for behavior_chunk in tqdm(behavior_chunks):
            start_time = time.time()

            batch_test_cases, batch_logs = self.generate_test_cases_chunk(behavior_chunk, verbose=verbose)
            for behavior_dict, test_cases, log in zip(behavior_chunk, batch_test_cases, batch_logs):
                behavior_id = behavior_dict['BehaviorID']

                if behavior_id in all_test_cases:
                    all_test_cases[behavior_id].append(test_cases)
                else:
                    all_test_cases[behavior_id] = [test_cases]

                if behavior_id in all_logs:
                    all_logs[behavior_id].append(log)
                else:
                    all_logs[behavior_id] = [log]

            if verbose:
                print(f"Time elapsed (s): {time.time() - start_time}")

        return all_test_cases, all_logs

    def generate_test_cases_single_behavior(self, behavior_dict, num_generate=1, verbose=False):
        """
        Generates test cases for a specific behavior. This is the main method that should be implemented by subclasses.

        This method can return a single test case and logs, or a list of test cases and logs. We check the type of the
        output and handle it accordingly in the `generate_test_cases` method. Note that this can cause issues if an
        individual test case and individual log are both lists already and have the same length. This can only happen
        for multimodal attacks, so it is recommended to wrap test cases and logs in lists for multimodal attacks.

        :param behavior_dict: a behavior dictionary specifying the behavior to generate test cases for
        :param num_generate: the number of test cases to generate (for attacks that can generate multiple test cases at once)
        :param verbose: whether to print progress
        :return: (list(test_cases), list(logs)) OR (test_case, logs); a list of test cases and logs for the behavior, or a single test case and logs
        """
        raise NotImplementedError

    def generate_test_cases_chunk(self, behavior_dicts, verbose=False):
        """
        :param behavior_dicts: a list of behavior dictionaries specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: (list(test_cases), list(logs)) OR (test_case, logs); a list of test cases and logs for the behavior, or a single test case and logs
        """

        behavior_strs = [b['Behavior'] for b in behavior_dicts]
        context_strs = [b['ContextString'] for b in behavior_dicts]
        #target_strs = [self.behavior_id_to_target.get(b['BehaviorID'], "Sure, here") for b in behavior_dicts]
        target_strs = [self.behavior_id_to_target.get(b['BehaviorID'], "Certainly!") for b in behavior_dicts]

        test_cases, _, logs_merged, _ = self.pgd.run(
            behavior_strs=behavior_strs, context_strs=context_strs, target_strs=target_strs, verbose=verbose)

        logs = []
        for i in range(len(behavior_dicts)):
            logs.append({
                k_o: {k_i: v_i[i].tolist()for k_i, v_i in v_o.items() if isinstance(v_i, np.ndarray)}
                for k_o, v_o in logs_merged.items()})

        return test_cases, logs

    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        run_id = f"_{run_id}" if run_id else ""
        return os.path.join(save_dir, 'test_cases_individual_behaviors', behavior_id, f"{file_type}{run_id}.json")

    def save_test_cases(self, save_dir, test_cases, logs=None, method_config=None, run_id=None):
        for behavior_id in test_cases.keys():
            test_cases_individual_behavior = {
                behavior_id: test_cases[behavior_id]}
            if logs is not None:
                logs = {behavior_id: logs.get(behavior_id, [])}
            self.save_test_cases_single_behavior(save_dir, behavior_id, test_cases_individual_behavior,
                                                 logs, method_config=method_config, run_id=run_id)

    def save_test_cases_single_behavior(
            self, save_dir, behavior_id, test_cases, logs=None, method_config=None, run_id=None):
        if save_dir is None:
            return

        test_cases_save_path = self.get_output_file_path(save_dir, behavior_id, 'test_cases', run_id)
        logs_save_path = self.get_output_file_path(save_dir, behavior_id, 'logs', run_id)
        method_config_save_path = self.get_output_file_path(save_dir, behavior_id, 'method_config', run_id)

        os.makedirs(os.path.dirname(test_cases_save_path), exist_ok=True)
        if test_cases is not None:
            with open(test_cases_save_path, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=4)

        if logs is not None:
            with open(logs_save_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4)

        if method_config is not None:
            # mask token or api_key
            self._replace_tokens(method_config)
            with open(method_config_save_path, 'w', encoding='utf-8') as f:
                method_config["dependencies"] = {l.__name__: l.__version__ for l in self.default_dependencies}
                json.dump(method_config, f, indent=4)

    @staticmethod
    def merge_test_cases(save_dir):
        """
        Merges {save_dir}/test_cases_individual_behaviors/{behavior_id}/test_cases.json into {save_dir}/test_cases.json
        (Also merges logs.json files for ease of access)
        """
        test_cases = {}
        logs = {}
        assert os.path.exists(save_dir), f"{save_dir} does not exist"

        # Find all subset directories
        behavior_ids = [d for d in os.listdir(os.path.join(
            save_dir, 'test_cases_individual_behaviors'))]

        # Load all test cases from each subset directory
        for behavior_id in behavior_ids:
            test_cases_path = os.path.join(
                save_dir, 'test_cases_individual_behaviors', behavior_id, 'test_cases.json')
            if os.path.exists(test_cases_path):
                with open(test_cases_path, 'r') as f:
                    test_cases_part = json.load(f)
                for behavior_id in test_cases_part:
                    assert behavior_id not in test_cases, f"Duplicate behavior ID: {behavior_id}"
                test_cases.update(test_cases_part)

        # Load all logs from each subset directory
        for behavior_id in behavior_ids:
            # Assuming logs are saved in json format
            logs_path = os.path.join(
                save_dir, 'test_cases_individual_behaviors', behavior_id, 'logs.json')
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    logs_part = json.load(f)
                for behavior_id in logs_part:
                    assert behavior_id not in logs, f"Duplicate behavior ID: {behavior_id}"
                logs.update(logs_part)

        # Save merged test cases and logs
        test_cases_save_path = os.path.join(save_dir, 'test_cases.json')
        logs_save_path = os.path.join(save_dir, 'logs.json')

        if len(test_cases) == 0:
            print(f"No test cases found in {save_dir}")
            return
        else:
            with open(test_cases_save_path, 'w') as f:
                json.dump(test_cases, f, indent=4)
            print(f"Saved test_cases.json to {test_cases_save_path}")

        if len(logs) == 0:
            print(f"No logs found in {save_dir}")
            return
        else:
            with open(logs_save_path, 'w') as f:  # Assuming you want to save logs in json format
                json.dump(logs, f, indent=4)
            print(f"Saved logs.json to {logs_save_path}")


# =============== REINFORCE_PGD CLASS DEFINITION ============================== #
class REINFORCE_PGD(PGD):
    """Identical to PGD, but with REINFORCE objective forcibly enabled."""

    def __init__(self, *args, reinforce_config: dict = dict(), **kwargs):
        reinforce_config['strategy'] = 'reinforce'
        kwargs.setdefault('early_stop_key', 'target_reinforce_agg_reinforce_loss')
        kwargs.setdefault('target_weight', 0.)
        super().__init__(*args, reinforce_config=reinforce_config, **kwargs)


def split_into_chunks(behaviors: list, bs: int):
    """Splits array into chunks of approximately equal size."""
    if len(behaviors) <= bs:
        return [behaviors]
    chunks = np.array_split(np.arange(len(behaviors)), int(np.ceil(len(behaviors) / bs)))
    behaviors = [[behaviors[i] for i in chunk] for chunk in chunks]
    return behaviors
