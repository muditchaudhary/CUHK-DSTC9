import os
import json
import random
import logging
import sys

from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["[CLS]", "[SEP]","<bos>", "<eos>", "[PAD]", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]

class Bert(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        
        # Bert special tokens
        self.cls = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["cls_token"])
        self.sep= self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS['sep_token'])
        
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])

        # PAD modified
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        # dataset_walker.py
        #   self.logs: logs.json
        #   self.labels: labels.json

        ## if labels_file passed in, use the output of task1 (baseline.ktd.json)
        ## only has target: True / False
        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        
        # self.dialogs: list of dictionary
        #   for train_baseline:
        #       format: [{'id': xx, 'log': [{'speaker': xx, 'text': xx}, {...}], 'label': {'target': xx, 'knowledge': [{'domain': xx, 'entity_id': xx}]},}
        #                {...},
        #                {...}]
        #       e.g. self.dialogs[0] = {'id': 0, 'log': [{'speaker': 'U', 'text': 'Looking for a place to eat in the city center.'}], 'label': {'target': False}}

        ##  for run_baseline: 'label' only has 'target'
        ##      format: [{'id': int, 'log': [{'speaker': string, 'text': string}, {...}, {...}, 'label': {'target': True/False},}
        ##               {...},
        ##               {...}]
        self.dialogs = self._prepare_conversations()

        # knowledge_reader.py
        #   self.knowledge: knowledge.json
        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)


        # self.snippets: dictionary
        #   format: {key: value}
        #   key: 'domain'
        #   value: list, tokenized knowledge, str(self.knowledge_sep_token).join([domain, name]), up to self.args.knowledge_max_tokens
        self.knowledge, self.snippets = self._prepare_knowledge()
        print("# of snippets = ",len(self.snippets.keys()))
        print("self.snippets: \n", self.snippets)
        print('\n\n')

        self._create_examples()
    
    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        status = 0
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                # should not have this part!!
                if "response" in label:
                    status = 1
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        print("dialog length = ", len(tokenized_dialogs))
        if status: print("Wrong!! It has response in label.json!! \n")
        else: print("No response in label.json\n")
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        # self.knowledge_docs: list of dictionaries
        # self.knowledge_docs = self.knowledge_reader.get_doc_list()
        self.knowledge_docs = self.knowledge_reader.get_domains()

        tokenized_snippets = dict()
        for snippet in self.knowledge_docs:
            key = "{}".format(snippet["domain"])
            knowledge = snippet["domain"]
            # knowledge = self._knowledge_to_string(snippet["domain"], name=snippet["entity_name"] or "")
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        print("knowledge length = ", len(tokenized_snippets)) # 145 = 33 + 110 + 1 + 1
        return knowledge, tokenized_snippets
    
    def _knowledge_to_string(self, doc, name=""):
        return doc["body"]
    
    def _create_examples(self):
        logger.info("Creating examples")

        # self.examples: list of dictionary
        self.examples = []
        
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"] # True or False

            # target == false, for task2 & 3, ignore
            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue
            
            # history: 2d list of one dialog, tokenized dialog text (no speaker info., later will be added manually)
            #   format: [[1st tokenized text], [2nd tokenized text], ...]
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]

            # get response from label if exists
            ## no response for run_baseline (baseline.ktd.json)
            gt_resp = label.get("response", "")
            # tokenize response
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:] # max num of utterance

            # data.py
            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens) # max num of tokens

            if target: # for task2 & 3
                if "knowledge" not in label: 
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                    ## for run_baseline & ????? all validation set evaluation: ??? all dummy knowledge is the 1st knowledge snippet in knowledge.json
                    ## label has no meaning ?? 
                    label["knowledge"] = [self.knowledge_docs[0]]

                # knowledge: 1st knowledge snippet in labels.json or a dummy knowledge
                knowledge = label["knowledge"][0] 
                
                knowledge_key = "{}".format(knowledge["domain"])
                # find snippets with same entity as candidates
                prefix = "{}".format(knowledge["domain"])

                # knowledge_candidates: list of strings, find keys in self.snippets that have the same prefix as the knowledge_key
                #   format: [key, key, ...]
                #   Fixed!! one problem: if knowledge_key == 'hotel__1', except all knowledge snippets of hotel entity 1, 'hotel_10', 'hotel_11' ... will also be included.
                # knowledge_candidates = [cand for cand in self.snippets.keys() if cand.startswith(prefix)]
                knowledge_candidates = [
                    cand
                    for cand in self.snippets.keys() 
                    if "__".join(cand.split("__")[:-1]) == prefix
                ]
                if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                    # if there's not enough candidates during training, we just skip this example
                    if len(knowledge_candidates) < self.args.n_candidates:
                        continue
                
                ## for run_baseline: dummy knowledge, 1st knowledge snippet
                used_knowledge = self.snippets[knowledge_key] # used knowledge
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens] # tokenized used knowledge
            else: # no need to do task2 & 3
                knowledge_candidates = None
                used_knowledge = []

            self.examples.append({
                "history": truncated_history, # 2d list, list of tokenized texts
                "knowledge": used_knowledge, # tokenized used knowledge ## dummy knowledge for run_baseline
                "candidates": knowledge_candidates, # list of keys of knowledge snippets, negative sampling candidates ???
                "response": tokenized_gt_resp, 
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })
    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)

class KnowledgeSelectionDataset(Bert):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, domain):
        return ''.join([domain])
        # return join_str.join([name, doc["title"], doc["body"]])

    def __getitem__(self, index):
        # one example in self.examples
        # one dialog, one example
        
        # format of one example:
        # {
        #     "history": truncated_history,
        #     "knowledge": used_knowledge,
        #     "candidates": knowledge_candidates,
        #     "response": tokenized_gt_resp,
        #     "response_text": gt_resp,
        #     "label": label,
        #     "knowledge_seeking": target,
        #     "dialog_id": dialog_id
        # }
        example = self.examples[index]

        # one example, one inst
        # one inst, n_candidates 
        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }

        ## for val and run_baseline
        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            ## for run_baseline
            # if self.args.eval_all_snippets:
            #     candidates = list(self.snippets.keys())
            # ## for val of evaluating validation set
            # else:
            #     candidates = example["candidates"]

            candidates = list(self.snippets.keys())
        else:
            if self.args.negative_sample_method == "all":
                candidates = list(self.snippets.keys())
            elif self.args.negative_sample_method == "mix":
                candidates = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
            elif self.args.negative_sample_method == "oracle":
                candidates = example["candidates"]
            else: # although we have already checked for this, still adding this here to be sure
                raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)
        # print(index, len(candidates), len(self.snippets))


        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        # candidates knowledge snippets of the selected candidate_keys
        candidates = [self.snippets[cand_key] for cand_key in candidates]

        # only sample n_candidates - 1 for "train"
        if self.split_type == "train":
            candidates = self._shrink_label_cands(example["knowledge"], candidates) # len(candidates) = self.args.n_candidates (includes used knowledge)

        if example["knowledge"] not in candidates: 
            print("Error! example['knowledge'] not in candidates")
            print(candidate_keys)
            print(candidates)
            print(example["knowledge"])
        label_idx = candidates.index(example["knowledge"]) # get the index of used knowledge
            
        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["attention_mask"].append(instance["attention_mask"])
        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        # knowledge: 1d list, tokenized knowledge snippets
        # history: 2d list, list of tokenized dialog texts
        instance = {}

        sequence = [[self.cls]] + history # [[self.cls],[text1],[text2],[text3],...] 2d list

        sequence_with_speaker = [ # add speaker to each text
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]

        # [[self.cls], [self.speaker1 + text1], [self.speaker2 + text2], [self.speaker1 + text1], ... , [self.sep], [knowledge tokens], [self.sep]], 2d list
        sequence = [sequence[0]] + sequence_with_speaker + [[self.sep] + knowledge + [self.sep]]

        instance["input_ids"] = list(chain(*sequence)) # 1d list of tokens
        # instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        instance["token_type_ids"] = [0] + [0 for s in sequence_with_speaker for i in s] + [0] + [1 for i in knowledge] + [1]
        # print(len(instance["token_typr_ids"]), len(instance["input_ids"]))
        instance['attention_mask'] =  [1  for _  in instance["input_ids"]]

        return instance, sequence
    
    ## only for  split_type = "train"
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        # random sample self.args.n_candidates-1 from other knowledge snippets
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
        # add the used knowledge back
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]
        attention_mask =[m for ins in batch for m in ins["attention_mask"]]
        
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        # print("====================")
        # print("before padding: ")
        # print("input_ids: ",len(input_ids))
        # print("token_type_ids: ", len(token_type_ids))
        # print("label_idx: ", len(label_idx))
        # print("attention_mask: ",len(attention_mask))

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        attention_mask = torch.tensor(
            pad_ids(attention_mask, self.pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        label_idx = torch.tensor(label_idx)

        # print("after padding: ")
        # print("input_ids: ",input_ids.size())
        # print("token_type_ids: ", token_type_ids.size())
        # print("label_idx: ", label_idx.size())
        # print("attention_mask: ",attention_mask.size())
        # print("====================")

        return input_ids, token_type_ids, lm_labels, label_idx, attention_mask, data_info
