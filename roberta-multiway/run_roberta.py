# coding=utf-8                                                                                                           
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.                                       
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.                                                          
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization_roberta import RobertaTokenizer
from modeling import BertForSequenceClassification
from optimization import BertAdam
from file_utils import PYTORCH_PRETRAINED_ROBERTA_CACHE
from modeling_roberta import RobertaMultiwayMatch, RobertaForMultipleChoice
from run_multiway_att import * #SwagExample, DataProcessor, CommonsenseQaProcessor, convert_examples_to_features
import sys
sys.path.append(".apex")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--roberta_model", default=None, type=str, required=True,
                        help="Roberta pre-trained model selected in the list: roberta-base, "
                             "roberta-large, roberta-large-mnli. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")

    args = parser.parse_args()

    processors = {
        "commonsenseqa": CommonsenseQaProcessor,
    }

    num_labels_task = {
        "commonsenseqa":4,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                      
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    print("current task is " + str(task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        
        print(args.train_batch_size)
        print(args.gradient_accumulation_steps)
        print(args.num_train_epochs)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    if args.load_model:
        output_model_file = os.path.join('C:/Users/14804/models/roberta-large-uncased', "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                     state_dict=model_state_dict)
    else:
        model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                 cache_dir=PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_{}'.format(
                                                     args.local_rank))
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
        
    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        model, optimizer = amp.initialize(model, optimizer, opt_level= "O1")

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    tr_acc = 0.0
    
    best_eval_accuracy = 0.0

    if args.do_train:
        train_features = convert_examples_to_roberta_features(train_examples, tokenizer,
                                                      args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(train_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(train_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(train_features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                   all_doc_len, all_ques_len, all_option_len)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        # Save a trained model                                                                                  
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self      
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_loss, tr_acc = 0.0, 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
                loss, logit = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    doc_len=doc_len, ques_len=ques_len, option_len=option_len, labels=label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.                                        

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses                                       
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                logit = logit.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tr_acc += accuracy(logit, label_ids)
                 
                if nb_tr_examples % 600 == 0:
                    print("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                    print("current train accuracy is %s" % (tr_acc / float(nb_tr_examples)))
                    
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_roberta_features(eval_examples, tokenizer,
                                                                     args.max_seq_length, True)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
                all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
                all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                          all_doc_len, all_ques_len, all_option_len)
                # Run prediction for full data                                                                           
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                for input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len \
                        in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    doc_len = doc_len.to(device)
                    ques_len = ques_len.to(device)
                    option_len = option_len.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids,
                                                      attention_mask=input_mask, doc_len=doc_len, ques_len=ques_len,
                                                      option_len=option_len, labels=label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_accuracy = eval_accuracy / nb_eval_examples
                print("the current eval accuracy is: " + str(eval_accuracy))
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy

                    if args.do_train:
                        torch.save(model_to_save.state_dict(), output_model_file)
                model.train()

    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)

    model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                 state_dict=model_state_dict)
    model.to(device)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.do_train:
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
            
        eval_features = convert_examples_to_roberta_features(eval_examples, tokenizer,
                                                             args.max_seq_length, False)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                  all_doc_len, all_ques_len, all_option_len)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_pred_labels = []
        all_anno_labels = []
        all_logits = []
        
        for input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            doc_len = doc_len.to(device)
            ques_len = ques_len.to(device)
            option_len = option_len.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                              doc_len=doc_len, ques_len=ques_len, option_len=option_len,
                                              labels=label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            output_labels = np.argmax(logits, axis=1)
            all_pred_labels.extend(output_labels.tolist())
            all_logits.extend(list(logits))
            all_anno_labels.extend(list(label_ids))
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        print("the eval accuracy is: " + str(eval_accuracy))
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'best_eval_accuracy': best_eval_accuracy,
                  'global_step': global_step}
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for i in range(len(all_pred_labels)):
                writer.write(str(i) + "\t" + str(all_anno_labels[i]) + "\t" +
                             str(all_pred_labels[i]) + "\t" + str(all_logits[i]) + "\n")

                
if __name__ == "__main__":
    main()
    
    
'''# coding=utf-8                                                                                                           
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.                                       
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.                                                          
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization_roberta import RobertaTokenizer
from modeling import BertForSequenceClassification
from optimization import BertAdam
from file_utils import PYTORCH_PRETRAINED_ROBERTA_CACHE
from modeling_roberta import RobertaMultiwayMatch, RobertaForMultipleChoice
from run_multiway_att import * #SwagExample, DataProcessor, CommonsenseQaProcessor, convert_examples_to_features
import sys
sys.path.append(".apex")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--roberta_model", default=None, type=str, required=True,
                        help="Roberta pre-trained model selected in the list: roberta-base, "
                             "roberta-large, roberta-large-mnli. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")

    args = parser.parse_args()

    processors = {
        "commonsenseqa": CommonsenseQaProcessor,
    }

    num_labels_task = {
        "commonsenseqa":4,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                      
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    print("current task is " + str(task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    if args.load_model:
        output_model_file = os.path.join('/content/drive/My Drive/roberta-large-uncased', "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                     state_dict=model_state_dict)
    else:
        model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                 cache_dir=PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_{}'.format(
                                                     args.local_rank))
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
        
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            from apex import am
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    tr_acc = 0.0
    
    best_eval_accuracy = 0.0

    if args.do_train:
        train_features = convert_examples_to_roberta_features(train_examples, tokenizer,
                                                      args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(train_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(train_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(train_features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                   all_doc_len, all_ques_len, all_option_len)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        # Save a trained model                                                                                  
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self      
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_loss, tr_acc = 0.0, 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
                loss, logit = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    doc_len=doc_len, ques_len=ques_len, option_len=option_len, labels=label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.                                        

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses                                       
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                logit = logit.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tr_acc += accuracy(logit, label_ids)
                 
                if nb_tr_examples % 600 == 0:
                    print("current train loss is %s" % (tr_loss / float(nb_tr_steps)))
                    print("current train accuracy is %s" % (tr_acc / float(nb_tr_examples)))
                    
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_roberta_features(eval_examples, tokenizer,
                                                                     args.max_seq_length, True)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
                all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
                all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                          all_doc_len, all_ques_len, all_option_len)
                # Run prediction for full data                                                                           
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                for input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len \
                        in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    doc_len = doc_len.to(device)
                    ques_len = ques_len.to(device)
                    option_len = option_len.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids,
                                                      attention_mask=input_mask, doc_len=doc_len, ques_len=ques_len,
                                                      option_len=option_len, labels=label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_accuracy = eval_accuracy / nb_eval_examples
                print("the current eval accuracy is: " + str(eval_accuracy))
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy

                    if args.do_train:
                        torch.save(model_to_save.state_dict(), output_model_file)
                model.train()

    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)

    model = RobertaMultiwayMatch.from_pretrained(args.roberta_model,
                                                 state_dict=model_state_dict)
    model.to(device)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.do_train:
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
            
        eval_features = convert_examples_to_roberta_features(eval_examples, tokenizer,
                                                             args.max_seq_length, False)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                  all_doc_len, all_ques_len, all_option_len)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_pred_labels = []
        all_anno_labels = []
        all_logits = []
        
        for input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            doc_len = doc_len.to(device)
            ques_len = ques_len.to(device)
            option_len = option_len.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                              doc_len=doc_len, ques_len=ques_len, option_len=option_len,
                                              labels=label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            output_labels = np.argmax(logits, axis=1)
            all_pred_labels.extend(output_labels.tolist())
            all_logits.extend(list(logits))
            all_anno_labels.extend(list(label_ids))
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        print("the eval accuracy is: " + str(eval_accuracy))
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'best_eval_accuracy': best_eval_accuracy,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            for i in range(len(all_pred_labels)):
                writer.write(str(i) + "\t" + str(all_anno_labels[i]) + "\t" +
                             str(all_pred_labels[i]) + "\t" + str(all_logits[i]) + "\n")
        
if __name__ == "__main__":
    main()
'''
