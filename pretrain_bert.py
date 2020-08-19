# coding=utf-8
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

"""Pretrain BERT"""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.data.bert_dataset import build_train_valid_test_datasets
from megatron.model import BertModel
from megatron.training import pretrain
from megatron.utils import reduce_losses


def model_provider():
    """Build the model."""

    print_rank_0('building BERT model ...')

    model = BertModel(
        num_tokentypes=2,
        add_binary_head=True,
        parallel_output=True)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask \
        = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model. lm_labels
    #for _ in range(1):
    #    model(tokens, padding_mask, tokentype_ids=types,lm_labels=lm_labels)

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            lm_loss_, sop_logits = model(tokens, padding_mask,
                                         tokentype_ids=types,
                                         lm_labels=lm_labels)
    exit()

    sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                               sentence_order.view(-1),
                               ignore_index=-1)

    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss + sop_loss

    reduced_losses = reduce_losses([lm_loss, sop_loss])

    return loss, {'lm loss': reduced_losses[0], 'sop loss': reduced_losses[1]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    import os
    import re
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    node_list = str(os.environ['SLURM_NODELIST'])
    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
    port = "23456"
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    torch.distributed.init_process_group("nccl", init_method=init_method,
                                         world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
