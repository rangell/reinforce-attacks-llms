import os
import random

import numpy as np
import torch
from sacred import Experiment
import seml
import wandb

experiment = Experiment()

@experiment.config
def default_seml_config():
    overwrite = None
    db_collection = None
    # db_collection can be set externally via sacred overrides (even though this code admitably looks a bit weird)
    if db_collection is not None:
        experiment.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@experiment.capture(prefix='wandb')
def wandb_initialize(
        _config, id: str | None, entity: str | None, project: str | None,
        group: str | None, mode: str | None, name: str | None,
        tags: list[str] | None, dir: os.PathLike | None,
        metrics: dict | None, enable: bool = True) -> wandb.sdk.wandb_run.Run | None:
    """ Initializes a wandb run. 

    Returns:
        Run: the wandb run
    """
    if not enable:
        return None

    if name is None:
        try:
            db_collection = _config['db_collection']
            run_id = _config['overwrite']
            name = os.path.join(str(db_collection), str(run_id))
        except KeyError as e:
            print('key error for wandb name: ', e)
            import time
            name = "dev" + str(time.time_ns())
            group = "dev"

    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    wandb_run = wandb.init(
        config=_config,
        id=id,
        entity=entity,
        project=project,
        group=group,
        mode=mode,
        name=name,
        tags=tags,
        dir=dir,
        resume=(mode == "online") and "allow")

    if metrics is not None:
        for key, aggs in metrics.items():
            wandb_run.define_metric(key, summary=aggs)

    return wandb_run


@experiment.capture()
def manual_seed(seed: int | None):
    """Seed all RNGs manually without reusing the same seed."""
    root_ss = np.random.SeedSequence(seed)

    num_rngs = 4
    if torch.cuda.is_available():
        num_rngs += torch.cuda.device_count()
    std_ss, np_ss, npg_ss, pt_ss, *cuda_ss = root_ss.spawn(num_rngs)

    # Python uses a Mersenne twister with 624 words of state, so we provide enough seed to
    # initialize it fully
    random.seed(std_ss.generate_state(624).tobytes())

    # We seed the global RNG anyway in case some library uses it internally
    np.random.seed(int(npg_ss.generate_state(1, np.uint32)))

    if torch.cuda.is_available():

        def lazy_seed_cuda():
            for i in range(torch.cuda.device_count()):
                device_seed = int(cuda_ss[i].generate_state(1, np.uint64))
                torch.cuda.default_generators[i].manual_seed(device_seed)

        torch.random.default_generator.manual_seed(
            int(pt_ss.generate_state(1, np.uint64)))
        torch.cuda._lazy_call(lazy_seed_cuda)

    # It is best practice not to use numpy's global RNG, so we instantiate one
    rng = np.random.default_rng(np_ss)

    seed = root_ss.entropy  # type: ignore
    return seed, rng