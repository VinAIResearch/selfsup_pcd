# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# # Change dataloader multiprocess start method to anything not fork
import open3d as o3d
import torch.multiprocessing as mp


try:
    mp.set_start_method("forkserver")  # Reuse process created
except RuntimeError:
    pass

import json
import logging
import os
import random
import sys

import hydra

# Torch packages
import torch
from easydict import EasyDict as edict
from lib import distributed_utils
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

# Train deps
from lib.test import test
from lib.train import train
from lib.utils import count_parameters, get_torch_device, load_state_with_same_shape
from models import load_model, load_wrapper
from omegaconf import OmegaConf
from torch.serialization import default_restore_location


def setup_logging(config):
    ch = logging.StreamHandler(sys.stdout)
    if config.distributed.distributed_world_size > 1 and config.distributed.distributed_rank > 0:
        logging.getLogger().setLevel(logging.WARN)
    else:
        logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format=os.uname()[1].split(".")[0] + " %(asctime)s %(message)s", datefmt="%m/%d %H:%M:%S", handlers=[ch]
    )


def main(config, init_distributed=False):

    if not torch.cuda.is_available():
        raise Exception("No GPUs FOUND.")

    # setup initial seed
    torch.cuda.set_device(config.distributed.device_id)
    torch.manual_seed(config.misc.seed)
    torch.cuda.manual_seed(config.misc.seed)

    device = config.distributed.device_id
    distributed = config.distributed.distributed_world_size > 1

    if init_distributed:
        config.distributed.distributed_rank = distributed_utils.distributed_init(config.distributed)

    setup_logging(config)

    logging.info("===> Configurations")
    # logging.info(config.pretty())

    DatasetClass = load_dataset(config.data.dataset)

    logging.info("===> Initializing dataloader")
    if config.train.is_train:
        train_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.train.train_phase,
            num_workers=config.data.num_workers,
            augment_data=True,
            shuffle=True,
            repeat=True,
            batch_size=config.data.batch_size,
            limit_numpoints=config.data.train_limit_numpoints,
        )

        val_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.data.num_val_workers,
            phase=config.train.val_phase,
            augment_data=False,
            shuffle=True,
            repeat=False,
            batch_size=config.data.val_batch_size,
            limit_numpoints=False,
        )

        if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3  # RGB color

        num_labels = train_data_loader.dataset.NUM_LABELS

    else:

        test_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.data.num_workers,
            phase=config.train.test_phase,
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=config.data.test_batch_size,
            limit_numpoints=False,
        )

        if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3  # RGB color

        num_labels = test_data_loader.dataset.NUM_LABELS

    logging.info("===> Building model")
    NetClass = load_model(config.net.model)
    if config.net.wrapper_type == None:
        model = NetClass(num_in_channel, num_labels, config)
        logging.info("===> Number of trainable parameters: {}: {}".format(NetClass.__name__, count_parameters(model)))
    else:
        wrapper = load_wrapper(config.net.wrapper_type)
        model = wrapper(NetClass, num_in_channel, num_labels, config)
        logging.info(
            "===> Number of trainable parameters: {}: {}".format(
                wrapper.__name__ + NetClass.__name__, count_parameters(model)
            )
        )

    logging.info(model)
    print("Check: ", config.net.weights)
    if config.net.weights == "modelzoo":  # Load modelzoo weights if possible.
        logging.info("===> Loading modelzoo weights")
        model.preload_modelzoo()

    # Load weights if specified by the parameter.
    elif config.net.weights.lower() != "none":
        logging.info("===> Loading weights: " + config.net.weights)
        # state = torch.load(config.weights)
        state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, "cpu"))

        if "state_dict" in state.keys():
            state_key_name = "state_dict"
        elif "model_state" in state.keys():
            state_key_name = "model_state"
        else:
            raise NotImplementedError

        if config.net.weights_for_inner_model:
            model.model.load_state_dict(state["state_dict"])
        else:
            if config.train.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state[state_key_name])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state["state_dict"])

    model = model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            bucket_cap_mb=config.distributed.bucket_cap_mb,
        )

    if config.train.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


@hydra.main(config_path="config", config_name="default.yaml")
def cli_main(config):
    # load the configurations
    if config.train.resume:
        resume_config = OmegaConf.load(os.path.join(config.train.resume, "config.yaml"))
        resume_config.train.resume = config.train.resume
        config = resume_config

    # if config.distributed.distributed_init_method is None:
    #   distributed_utils.infer_init_method(config.distributed)

    if config.distributed.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not config.distributed.distributed_no_spawn:
            # TODO(Ji) _pickle.PicklingError: Can't pickle <function distributed_main at 0x7f78e2caab00>: attribute lookup distributed_main on __main__ failed
            start_rank = config.distributed.distributed_rank
            config.distributed.distributed_rank = None  # assign automatically
            mp.spawn(
                fn=distributed_main,
                args=(config, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(config.distributed.device_id, config)

    elif config.distributed.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert config.distributed.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config.distributed.distributed_init_method = "tcp://localhost:{port}".format(port=port)
        config.distributed.distributed_rank = None  # set based on device id

        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(config,),
            nprocs=config.distributed.distributed_world_size,
        )
    else:
        # single GPU training
        main(config)


def distributed_main(i, config, start_rank=0):
    config.distributed.device_id = i
    if config.distributed.distributed_rank is None:  # torch.multiprocessing.spawn
        config.distributed.distributed_rank = start_rank + i
    main(config, init_distributed=True)


if __name__ == "__main__":
    __spec__ = None
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # start
    cli_main()
