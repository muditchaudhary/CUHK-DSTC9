import json
from tqdm import tqdm
import torch
import os
from load_database import load_database
import random
from fuzzywuzzy import process
import numpy as np
from api_src.surfMatch import *
from entity_classifier import extract_entity

id_entity_name = ['name']
from domain_classifier import extract_domain

domain_set = ['hotel', 'restaurant', 'train', 'taxi', 'attraction']
entity_names_for_all_domains = {}


class DSTC_NLI_Dataset_for_Pipe(torch.utils.data.Dataset):
    def __init__(self, args, split_type, tokenizer=None, num_neg_candidates=3,
                 hypothesis_template="The user is asking about {}.", System_response_prefix="",
                 User_response_prefix="", if_ood=False):
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
        self.num_neg_candidates = num_neg_candidates
        self.utterance_window = args.utterance_window
        self.split_type = split_type
        self.hypothesis_template = hypothesis_template
        self.System_response_prefix = System_response_prefix
        self.User_response_prefix = User_response_prefix
        self.premise_hypothesis_examples = []
        self.gt_labels = []
        #self.domain_classifier = domain_classifier.domain_classifier()
        self.knowledge_seeking_gts = []
        self.if_ood = if_ood

        self.get_names_for_all_domains() # Consolidate entity names for all domains from the database

        self._sample_and_create_examples()

        assert len(self.premise_hypothesis_examples) == len(self.premise_hypothesis_examples), "Label-Examples mismatch"
        print('len of examples', len(self.premise_hypothesis_examples))
        for example in self.premise_hypothesis_examples:
            if isinstance(example, str):
                print('it is tuple', example)

    def _sample_and_create_examples(self):
        """
        Samples and creates the premise-hypothesis examples.
        :return: None
        :rtype:
        """
        global domain_set
        if self.if_ood == False:
            with open(os.path.join(self.dataroot, self.split_type, "logs.json")) as logs_file:
                logs = json.load(logs_file)
            # with open(os.path.join(self.dataroot, self.split_type, "labels.json")) as labels_file:
            #    labels = json.load(labels_file)
            with open(os.path.join(self.dataroot, "knowledge.json")) as knowledge_file:
                self.knowledge = json.load(knowledge_file)

        #database_dialog, database_label = load_database.load_database(self.extracted_data)
        for log in tqdm(logs, disable=False):
            example = {}
            if (len(log) > 2 and self.utterance_window == 3):
                premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
                          log[-2]['text'] + ". " + self.User_response_prefix + log[-1]['text']
            elif (len(log) > 1):
                premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + log[-1][
                    'text']
            else:
                premise = self.User_response_prefix + log[-1]['text']

            example['logs'] = log
            example['choices'] = []
            example['choice_types'] = []
            domain = extract_domain(premise=premise, labels_list=domain_set)
            domain = domain["Pred_domain"]
            name = extract_entity(log=log, domain=domain, entity_list=entity_names_for_all_domains)
            entity_id = self.find_knowledge_entity_id(name, domain)
            database_examples = self.get_database_knowledge_for_knowledge_example(premise, domain, entity_id)
            example['choices'].extend(database_examples)
            example['choice_types'].extend([False] * len(database_examples))
            # torch.nn.NLLLoss requires only class idx
            # logits are in order [contradiction, neutral, entailment]
            remaining_knowledge = [knowledge_snippets['body'] for know_id, knowledge_snippets in
                                   self.knowledge[domain][entity_id]['docs'].items()]

            if self.split_type == 'train':
                remaining_knowledge = random.sample(remaining_knowledge,
                                                    min(self.num_neg_candidates, len(remaining_knowledge)))

            example['choices'].extend(remaining_knowledge)
            example['choice_types'].extend([True] * len(remaining_knowledge))
            self.premise_hypothesis_examples.append(example)
        #print('len of database', len(database_dialog))

    def get_database_knowledge_for_knowledge_example(self, premise, domain, entity_id):
        with open("./db/"+domain + '_db.json', 'r') as f:
            domain_database = json.load(f)
        candidates = []
        if entity_id == '*':
            if isinstance(domain_database[0], dict):
                random_object = random.sample(domain_database, 1)[0]
                for entity, value in random_object.items():
                    candidates.append(self.to_sentence({entity: value}, domain, domain))

            else:
                for entity, value in domain_database.items():
                    candidates.append(self.to_sentence({entity: value}, domain, domain))
        else:
            name = self.knowledge[domain][entity_id]['name']
            if isinstance(domain_database[0], dict):
                name_entity = None
                for entity in domain_database[0].keys():
                    if entity in id_entity_name:
                        name_entity = entity
                        break
                else:
                    raise ('There is a new name_entity, that not in id_entity_name')
                for object_detail in domain_database:
                    if object_detail[name_entity].lower() == name.lower():

                        for entity, value in object_detail.items():
                            if entity != name_entity:
                                candidates.append(self.to_sentence({entity: value}, name, domain))
                        break
        if self.split_type == 'train':
            candidates = random.sample(candidates, min(self.num_neg_candidates, len(candidates)))

        return candidates

    def get_database_knowledge_for_database_example(self, premise, log, gt_label):
        domain = domain_cls(log)
        remain_entities = []
        neg_knowledge = []
        name = None
        if isinstance(domain_database[0], dict):
            if entity_names_for_all_domains[domain] != []:
                name = surfMatch.get_1ent(log, entity_names_for_all_domains[domain], known_domain=domain)

            entities = self.extract_entities_by_name(name, domain_database, domain)
            if name == None:
                name = domain
            for entity, value in entities.items():
                # if entity!=gt_entity and value!=name:
                # candidate=self.hypothesis_template.format(name,entity,value)
                candidate = self.to_sentence({entity: value}, name, domain)
                remain_entities.append(candidate)
        else:
            for entity, value in domain_database.items():  # the format of taxi_db need to be changed
                # if entity!=gt_entity and value!=name:
                # candidate=self.hypothesis_template.format(domain,entity,value)
                candidate = self.to_sentence({entity: value}, name, domain)
                remain_entities.append(candidate)

        if self.split_type == 'train':
            remain_entities = random.sample(remain_entities, min(self.num_neg_candidates, len(remain_entities)))

        database_examples = remain_entities
        remain_entities = []

        for name_id, object_detail in self.knowledge[domain].items():
            if name_id != '*':
                if name != None:
                    if object_detail['name'].lower() == name.lower():  # may use other method to compare
                        for knowledge_snippet_id, knowledge_snippet in object_detail['docs'].items():
                            candidate = knowledge_snippet['body']
                            remain_entities.append(candidate)
                else:
                    random_name_id = random.sample(list(self.knowledge[domain].keys()), 1)[0]

                    for knowledge_snippet_id, knowledge_snippet in self.knowledge[domain][random_name_id][
                        'docs'].items():
                        candidate = knowledge_snippet['body']
                        remain_entities.append(candidate)
                    break
            else:
                for knowledge_snippet_id, knowledge_snippet in object_detail['docs'].items():
                    candidate = knowledge_snippet['body']
                    remain_entities.append(candidate)
        else:
            if remain_entities == []:
                random_name_id = random.sample(list(self.knowledge[domain].keys()), 1)[0]
                for knowledge_snippet_id, knowledge_snippet in self.knowledge[domain][random_name_id]['docs'].items():
                    candidate = knowledge_snippet['body']
                    remain_entities.append(candidate)

        if self.split_type == 'train':

            remain_entities = random.sample(remain_entities, min(self.num_neg_candidates, len(remain_entities)))
        knowledge_examples = remain_entities
        return database_examples, knowledge_examples

    def to_sentence(self, dic, name, dname):
        sentence = ''
        for k in dic.keys():
            value = dic[k]
            if type(value) != str:
                if isinstance(value, dict):
                    value = list(value.keys())[0]
                if isinstance(value, list):
                    tmp_value = ''
                    for item in value:
                        tmp_value += (str(item) + ' ')
                    value = tmp_value
                value = str(value)

            if dic[k] == 'no':
                sentence += (name + ' has no ' + k + '. ')
            elif dic[k] == 'yes':
                sentence += (name + ' has ' + k + '. ')
            elif k == 'beginning':
                sentence += ('You are looking for the ' + dname + ' and planning to book it.')
            elif k == 'stars':
                sentence += (name + ' has stars of ' + value + '. ')
            elif k == 'leaveAt':
                sentence += (name + ' leave at ' + value + '. ')
            elif k == 'arriveBy':
                sentence += (name + ' arrive by ' + value + '. ')
            elif k == 'destination':
                sentence += (name + '\'s destination is ' + value + '. ')
            elif k == 'departure':
                sentence += (name + '\'s departure is ' + value + '. ')
            elif k == 'day':
                sentence += ('The day is ' + value + '. ')
            elif k == 'type':
                sentence += ('The type is ' + dic[k] + '. ')
            elif k == 'area':
                sentence += ('The area of ' + name + ' is in the ' + value + '. ')
            elif k == 'name':
                sentence += ('The name of this ' + dname + ' is ' + value + '. ')
            elif k == 'ending':
                sentence += 'You are welcome. Good bye.'
            else:
                sentence += ('The ' + k + ' of ' + name + ' is ' + value + '. ')
        if isinstance(sentence, tuple):
            print('it is a tuple', dic, name, dname)
        if sentence == '':
            print('it is  none,', name, domain)
            exit()
        return sentence

    def extract_entities_by_name(self, name, domain_database, domain):
        if name != None:
            for object_detail in domain_database:
                for entity, value in object_detail.items():
                    if entity in id_entity_name:
                        if value.lower() == name.lower():
                            return object_detail
            else:
                print('name', name, domain)
                print()
                print('there is no such name')
                #                 exit()
                return random.sample(domain_database, 1)[0]
        else:
            return domain_database[0]

    def fliter_name(self, name, domain):
        if name == None:
            return None
        if '|' in name:
            name = name[:name.index('|')]
        if name in self.entity_name_start_with_the[domain]:
            return 'the ' + name
        else:
            return name

    def find_knowledge_entity_id(self, name, domain):
        knowledge_snippets = self.knowledge
        if name == None:
            if list(knowledge_snippets[domain].keys())[0] == '*':
                return '*'
            else:
                return random.sample(list(knowledge_snippets[domain].keys()), 1)[0]
        else:
            for entity_id, details in knowledge_snippets[domain].items():
                if details['name'] != None:
                    if details['name'].lower() == name.lower():
                        return entity_id
                    else:
                        continue
                else:
                    return '*'
            else:
                print('this is the name', name)
                return random.sample(list(knowledge_snippets[domain].keys()), 1)[0]
                raise ('cannot find the entity_id')

    def preprocess_name(self):
        print('i am here')
        self.entity_name_start_with_the = {}
        for domain in domain_set:
            self.entity_name_start_with_the[domain] = []
            with open(domain + '_db.json', 'r') as f:
                database_of_domain = json.load(f)
            entity_name = None
            if isinstance(database_of_domain[0], dict):
                for entity in database_of_domain[0].keys():
                    if entity in id_entity_name:
                        entity_name = entity
            else:
                continue
            if entity_name == None:
                continue
            for object_detail in database_of_domain:
                if object_detail[entity_name].lower().startswith('the '):
                    self.entity_name_start_with_the[domain].append(object_detail[entity_name].lower()[4:])

    def get_cand_entNames(self, domain_database):
        entity_name = None
        cand_entNames = []
        for entity in domain_database[0].keys():
            if entity in id_entity_name:
                entity_name = entity
        if entity_name == None:
            return None
        for object_detail in domain_database:
            cand_entNames.append(object_detail[entity_name])
        return cand_entNames

    def get_names_for_all_domains(self):
        """
        Gets entity names from the <domain>_db.json for each domain
        """
        for domain in domain_set:
            entity_names_for_all_domains[domain] = []
            with open("./db/"+ domain + '_db.json') as f:
                domain_database = json.load(f)
            entity_name = None
            for entity in domain_database[0].keys():
                if entity in id_entity_name:
                    entity_name = entity
            if entity_name == None:
                continue
            for object_detail in domain_database:
                entity_names_for_all_domains[domain].append(object_detail[entity_name])

    def name_cls(self, log, domain):

        with open(domain + '_db.json', 'r') as f:
            domain_dataset = json.load(f)

        log_text = ''
        log_type = 0
        log_len = min(len(log), 5)
        if log_len % 2 == 1:
            log_type = 0
        else:
            log_type = 1
        for log_idx in log[-log_len:]:
            if log_type == 0:
                log_text += self.User_response_prefix + log['text']
                log_type = 1
            else:
                log_text += self.System_response_prefix + log['text']
                log_type = 0

        name_set = []
        if isinstance(domain_dataset[0], dict):
            for object_detail in domain_dataset:
                for entity, value in object_detail.items():
                    if entity in id_entity_name:
                        name_set.append(value)
                        break
                else:
                    raise ('There is no name in this domain')

        ratios = []
        for name in name_set:
            ratios = process.extract(name, log_text)
        return name[np.argmax(ratios)]

    def __len__(self):
        return (len(self.premise_hypothesis_examples))

    def __getitem__(self, item):
        sample = self.premise_hypothesis_examples[item]
        return sample
