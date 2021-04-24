# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT2"""
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.global_vars import set_use_wandb, get_use_wandb
from megatron.model import GPT2Model, GPT2ModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, is_local_main, local_rank, get_wandb_api_key, neox_args
from megatron.utils import reduce_losses
from megatron.fp16 import fp32_to_fp16

import wandb
from wandb import UsageError


KD_loss = nn.KLDivLoss(reduction='batchmean')

def kd_step(teacher: nn.Module,
            student: nn.Module,
            temperature: float,
            inputs: torch.tensor,
            optimizer: Optimizer):
    teacher.eval()
    student.train()
    
    with torch.no_grad():
        logits_t = teacher(inputs=inputs)
    logits_s = student(inputs=inputs)
    
    loss = KD_loss(input=F.log_softmax(logits_s/temperature, dim=-1),
                   target=F.softmax(logits_t/temperature, dim=-1))
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
            
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
        train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')
