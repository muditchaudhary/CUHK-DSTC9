import json
from tqdm import tqdm
import torch
import os


class DSTC_NLI_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split_type, tokenizer,
                 hypothesis_template="The user is asking about {}.", System_response_prefix="",
                 User_response_prefix=""):
        """
        :param args: Contains dataroot and labels
        :type args:
        :param split_type: Train, val, test
        :type split_type:
        :param tokenizer:
        :type tokenizer:
        :param hypothesis_template: Hypothesis template to be paired with the utterance (premise)
        :type hypothesis_template: str
        :param System_response_prefix: Prefix to the utterance from the system eg. "The systems says "
        :type System_response_prefix: str
        :param User_response_prefix: Prefix to the utterance from the user eg. "The user says "
        :type User_response_prefix: str
        """
        self.dataroot = args.dataroot
        self.max_length = args.max_length
        self.tokenizer = tokenizer
        self.utterance_window = args.utterance_window
        self.split_type = split_type
        self.hypothesis_template = hypothesis_template
        self.System_response_prefix = System_response_prefix
        self.User_response_prefix = User_response_prefix
        self.premise_hypothesis_examples = []
        self.gt_labels = []
        self.candidate_labels = ['hotel', 'restaurant', 'taxi', 'train', 'attraction', 'bus', 'hospital', 'police']

        self._sample_and_create_examples()
        assert len(self.gt_labels) == len(self.premise_hypothesis_examples), "Label-Examples mismatch"

    def _sample_and_create_examples(self):
        """
        Samples and creates the premise-hypothesis examples. Only creates the examples using the utterances
        whose domains are known.
        :return: None
        :rtype:
        """
        with open(os.path.join(self.dataroot, self.split_type, "logs.json")) as logs_file:
            self.logs = json.load(logs_file)

        with open(os.path.join(self.dataroot, self.split_type, "labels.json")) as labels_file:
            self.labels = json.load(labels_file)

        for log, label in tqdm(zip(self.logs, self.labels)):

            if label['target']:
                if (len(log) > 2 and self.utterance_window == 3):
                    premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
                              log[-2]['text'] + ". " + self.User_response_prefix + log[-3]['text']
                elif (len(log) > 1):
                    premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + \
                              log[-1]['text']
                else:
                    premise = self.User_response_prefix + log[-1]['text']

                domain = label['knowledge'][0]['domain']
                positive_hypothesis = self.hypothesis_template.format(domain)
                tokenized_positive_example = self.tokenizer(premise, positive_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                            , truncation=True)
                self.premise_hypothesis_examples.append(tokenized_positive_example)
                self.gt_labels.append(
                    torch.tensor([2]))
                # torch.nn.NLLLoss requires only class idx
                # logits are in order [contradiction, neutral, entailment]

                remaining_labels = [neg_domain for neg_domain in self.candidate_labels if neg_domain is not domain]
                for neg_label in remaining_labels:
                    neg_domain = neg_label
                    negative_hypothesis = self.hypothesis_template.format(neg_domain)
                    tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                                max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
                    self.premise_hypothesis_examples.append(tokenized_negative_example)
                    self.gt_labels.append(
                        torch.tensor([0]))  # logits are in order [contradiction, neutral, entailment]

    def __len__(self):
        return (len(self.gt_labels))

    def __getitem__(self, item):
        sample = self.premise_hypothesis_examples[item]
        label = self.gt_labels[item]

        return sample, label


class DSTC_NLI_DB_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split_type, tokenizer,
                 hypothesis_template="The user is asking about {}.", System_response_prefix="",
                 User_response_prefix=""):
        """
        :param args: Contains dataroot and labels
        :type args:
        :param split_type: Train, val, test
        :type split_type:
        :param tokenizer:
        :type tokenizer:
        :param hypothesis_template: Hypothesis template to be paired with the utterance (premise)
        :type hypothesis_template: str
        :param System_response_prefix: Prefix to the utterance from the system eg. "The systems says "
        :type System_response_prefix: str
        :param User_response_prefix: Prefix to the utterance from the user eg. "The user says "
        :type User_response_prefix: str
        """
        self.dataroot = args.dataroot
        self.max_length = args.max_length
        self.tokenizer = tokenizer
        self.utterance_window = args.utterance_window
        self.split_type = split_type
        self.hypothesis_template = hypothesis_template
        self.System_response_prefix = System_response_prefix
        self.User_response_prefix = User_response_prefix
        self.premise_hypothesis_examples = []
        self.gt_labels = []
        self.candidate_labels = ['hotel', 'restaurant', 'taxi', 'train', 'attraction', 'bus', 'hospital', 'police']

        self._sample_and_create_examples()
        assert len(self.gt_labels) == len(self.premise_hypothesis_examples), "Label-Examples mismatch"

    def _sample_and_create_examples(self):
        """
        Samples and creates the premise-hypothesis examples. Only creates the examples using the utterances
        whose domains are known.
        :return: None
        :rtype:
        """
        with open(os.path.join(self.dataroot, self.split_type, "logs.json")) as logs_file:
            self.logs = json.load(logs_file)

        with open(os.path.join(self.dataroot, self.split_type, "labels.json")) as labels_file:
            self.labels = json.load(labels_file)

        for log, label in tqdm(zip(self.logs, self.labels)):

            if label['target']:
                if (len(log) > 2 and self.utterance_window == 3):
                    premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
                              log[-2]['text'] + ". " + self.User_response_prefix + log[-3]['text']
                elif (len(log) > 1):
                    premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + \
                              log[-1]['text']
                else:
                    premise = self.User_response_prefix + log[-1]['text']

                domain = label['knowledge'][0]['domain']
                positive_hypothesis = self.hypothesis_template.format(domain)
                tokenized_positive_example = self.tokenizer(premise, positive_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                            , truncation=True)
                self.premise_hypothesis_examples.append(tokenized_positive_example)
                self.gt_labels.append(
                    torch.tensor([2]))
                # torch.nn.NLLLoss requires only class idx
                # logits are in order [contradiction, neutral, entailment]

                remaining_labels = [neg_domain for neg_domain in self.candidate_labels if neg_domain is not domain]
                for neg_label in remaining_labels:
                    neg_domain = neg_label
                    negative_hypothesis = self.hypothesis_template.format(neg_domain)
                    tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                                max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
                    self.premise_hypothesis_examples.append(tokenized_negative_example)
                    self.gt_labels.append(
                        torch.tensor([0]))  # logits are in order [contradiction, neutral, entailment]

    def __len__(self):
        return (len(self.gt_labels))

    def __getitem__(self, item):
        sample = self.premise_hypothesis_examples[item]
        label = self.gt_labels[item]

        return sample, label
