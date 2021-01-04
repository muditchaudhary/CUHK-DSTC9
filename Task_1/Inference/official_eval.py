# from fuzzywuzzy import process
import json
import numpy as np
import torch
from tqdm import trange, tqdm
import os
import logging
import json
from sklearn.metrics import precision_recall_fscore_support
from argparse import Namespace

from transformers import pipeline
import argparse
import official_dataset
# model="./runs_POD_version3/checkpoint-2400_new"
model="./runs_POD_version3/checkpoint-2540-tfexample-30000"
classifier = pipeline("zero-shot-classification", model=model)
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def evaluate(args,eval_dataset,outfile):
    args.eval_batch_size=1
    eval_sampler=SequentialSampler(eval_dataset)
    # Set eval batch size = 1
    eval_dataloader=DataLoader(eval_dataset,sampler=eval_sampler,batch_size=args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    gt_labels = []
    accuracy=0
    knowledge_detection_accuracy=0
    tp,fp,tn,fn=0,0,0,0
    examples=[]
    for batch in tqdm(eval_dataloader,desc='Official Evaluation'):
        # example,label=batch
        example=batch
#         print(example)
        logs=example['logs']
        example['choices']=[choice[0] for choice in example['choices']]
        choices=example['choices']
        choice_types=example['choice_types']
        print(choices)
        pred=classifier(logs[-1], choices)
        preds.append(pred )
        print('pred',pred['labels'][0])
        examples.append(example)
    print(preds)
    pred_types=[]
    pred_answers=[]
    final_answers={}
    
    for index,data in enumerate(zip(preds,examples)):
        pred,example=data[0],data[1]
        pred_target=example['choice_types'][example['choices'].index(pred['labels'][0])]
        choice=pred['labels'][0]
        final_answers[str(index)]={'log':example['logs'],'pred':bool(pred_target),'all_choice':pred['labels']}
        print(final_answers[str(index)])
    with open('test_final_answers.json','w') as f:
        json.dump(final_answers,f,indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--outfile", type=str, help="File to save predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # parser.add_argument('--labels', '--list', nargs='+', help='Labels', required=True)
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

    eval_dataset=official_dataset.DSTC_NLI_Dataset_for_Pipe(args=args,split_type='val', System_response_prefix="Assistant says ",
                                 User_response_prefix="User says ")
    evaluate(args,eval_dataset)