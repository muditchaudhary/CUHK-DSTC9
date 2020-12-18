import numpy as np
import torch
from dataset_with_pesudo_label import DSTC_NLI_Dataset
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (AdamW,
                          pipeline,
                          AutoModelForSequenceClassification,
                          AutoTokenizer)
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange, tqdm
import os
import logging
import json
from sklearn.metrics import precision_recall_fscore_support
from argparse import Namespace

logger = logging.getLogger(__name__)


def load_dataset(args, split_type, tokenizer, hypothesis_template="", System_response_prefix="",
                 User_response_prefix="",num_neg_candidates=2,if_ood=False):
    """
    :param args:
    :type args:
    :return: torch.utils.data.Dataset object
    :rtype:
    """

    dataset = DSTC_NLI_Dataset(args,  split_type=split_type,tokenizer=tokenizer, hypothesis_template=hypothesis_template, System_response_prefix=System_response_prefix,
                               User_response_prefix=User_response_prefix,num_neg_candidates=num_neg_candidates,if_ood=if_ood)

    return dataset


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """
    Training function
    :param args:
    :type args:
    :param train_dataset:
    :type train_dataset:
    :param eval_dataset:
    :type eval_dataset:
    :param model:
    :type model:
    :return:
    :rtype:
    """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_training_steps)

    if args.n_gpu > 1:
        # Multi-GPU training
        model = torch.nn.DataParallel(model)

    # Start training
    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epochs")
    trained_epoch=0
    for _ in train_iterator:
        local_steps = 0
        training_loss = 0.0
        trained_epoch+=1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            samples, labels = batch
            samples['input_ids'] = samples['input_ids'].squeeze(dim=1)
            samples['attention_mask'] = samples['attention_mask'].squeeze(dim=1)
            loss, _, _ = model(**samples, labels=labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps >1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()
            training_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=training_loss / local_steps)
        if trained_epoch==2 and False:
            results = evaluate(args, eval_dataset, model,global_step)
        if trained_epoch>=5:
            output_dir = os.path.join(args.output_dir, "{}-{}-tfexample-30000-version3".format("checkpoint", global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )

            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
            logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, training_loss / local_steps


def evaluate(args, eval_dataset, model,global_step):
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = 1

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
    )
    # Set eval batch size = 1

    if args.n_gpu > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    preds = []
    gt_labels = []
    num=0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         num+=1
#         if num==1000:
#             break
        with torch.no_grad():
            samples, label = batch
            samples['input_ids'] = samples['input_ids'].squeeze(dim=1)
            samples['attention_mask'] = samples['attention_mask'].squeeze(dim=1)
            loss, cls_logits, _ = model(**samples, labels=label)
            softmax = torch.nn.Softmax()
            cls_logits = softmax(cls_logits)
            preds.append(cls_logits.detach().cpu().numpy())
            eval_loss += loss.mean().item()
            gt_labels.append(label)
            
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    assert len(preds) == len(gt_labels)
    preds = np.concatenate(preds)
    preds = preds.argmax(axis=1)
    gt_labels = np.concatenate(gt_labels)
    accuracy = np.sum(preds == gt_labels) / len(preds)

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gt_labels, preds, average="micro")
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gt_labels, preds, average="macro")

    eval_result = {'model':str(global_step)+'30000',"loss": eval_loss, "accuracy": accuracy, "micro_precision": micro_precision,
                   "micro_recall": micro_recall,
                   "micro_f1": micro_f1, "macro_precision": macro_precision, "macro_recall": macro_recall,
                   "macro_f1": macro_f1}

    with open(os.path.join(eval_output_dir, "eval_results.txt"), 'a') as f:
        for key in eval_result.keys():
            f.write("%s = %s\n" % (key, str(eval_result[key])))
        f.write("-----\n\n")

    return eval_result


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)

    args.params = params
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    train_dataset = load_dataset(args=args, split_type='train', tokenizer=tokenizer,
                                 hypothesis_template="User is asking about {}.", System_response_prefix="Assistant says ",
                                 User_response_prefix="User says ",if_ood=False)
#     eval_dataset = load_dataset(args=args, split_type='val', tokenizer=tokenizer,
#                                 hypothesis_template="User is asking about {}.", System_response_prefix="Assistant says ",
#                                 User_response_prefix="User says ",if_ood=True)
    eval_dataset=None
    global_step, training_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, training_loss)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
        json.dump(params, jsonfile, indent=2)
    
#     eval_dataset = load_dataset(args=args, split_type='test', tokenizer=tokenizer,
#                                 hypothesis_template="User is asking about {}.", System_response_prefix="Assistant says ",
#                                 User_response_prefix="User says ",if_ood=True)
#     eval_result=evaluate(args, eval_dataset, model,'eval')
#     print(eval_result)

if __name__ == "__main__":
    main()
