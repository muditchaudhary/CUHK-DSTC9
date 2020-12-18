import json
from tqdm import tqdm
import torch
import os
import load_database
import random
from fuzzywuzzy import process
import numpy as np
import api_src.surfMatch as surfMatch
id_entity_name=['name']
domain_set=['hotel','restaurant','train','taxi']
class DSTC_NLI_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split_type, tokenizer, num_neg_candidates=8,
                hypothesis_template="The user is asking about {}.", System_response_prefix="",
                 User_response_prefix="",if_ood=False):
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
        self.num_neg_candidates=num_neg_candidates
        self.utterance_window = args.utterance_window
        self.split_type = split_type
        self.hypothesis_template = hypothesis_template
        self.System_response_prefix = System_response_prefix
        self.User_response_prefix = User_response_prefix
        self.premise_hypothesis_examples = []
        self.gt_labels = []
        self.preprocess_name()
        self.if_ood=if_ood
#         exit()
        # self.candidate_labels = ['hotel', 'restaurant', 'taxi', 'train']


        self._sample_and_create_examples()
        assert len(self.gt_labels) == len(self.premise_hypothesis_examples), "Label-Examples mismatch"

    def _sample_and_create_examples(self):
        """
        Samples and creates the premise-hypothesis examples. Only creates the examples using the utterances
        whose domains are known.
        :return: None
        :rtype:
        """
        global domain_set
        if self.if_ood==False:
            with open(os.path.join(self.dataroot, self.split_type, "logs.json")) as logs_file:
                logs = json.load(logs_file)

            with open(os.path.join(self.dataroot, self.split_type, "labels.json")) as labels_file:
                labels = json.load(labels_file)
            with open(os.path.join(self.dataroot,self.split_type,"database.json")) as extracted_data_file:
                self.extracted_data=json.load(extracted_data_file)
            with open(os.path.join(self.dataroot,"knowledge.json")) as knowledge_file:
                self.knowledge=json.load(knowledge_file)
            with open('data/train/new_labels_with_last_sentence_official.json','r') as f:
                self.new_labels=json.load(f)[:39000]
                self.num_data=len(self.new_labels)
                print('len of new label',self.new_labels[0])
            
        else:
            
            with open(os.path.join(self.dataroot, self.split_type,'ood', "logs-version3.json")) as logs_file:
                logs = json.load(logs_file)

            with open(os.path.join(self.dataroot, self.split_type,'ood', "labels-version3.json")) as labels_file:
                labels = json.load(labels_file)
            with open(os.path.join(self.dataroot,self.split_type,"database.json")) as extracted_data_file:
                self.extracted_data=json.load(extracted_data_file)
            with open(os.path.join(self.dataroot,"knowledge_old.json")) as knowledge_file:
                self.knowledge=json.load(knowledge_file)
#             with open('data/train/new_labels_with_last_sentence_official.json','r') as f:
            with open('data/train/ood/pseudo_labelled_dataset-version3.json','r') as f:
                self.new_labels=json.load(f)
                self.num_data=len(self.new_labels)
                print('len of new label',len(self.new_labels))
            if self.split_type=='train':
#                 domain_set=['restaurant','train']
                domain_set=['hotel','taxi']
#             if self.split_type=='Test':
#                 with open(os.path.join(self.dataroot,"test_knowledge.json")) as knowledge_file:
#                     self.knowledge=json.load(knowledge_file)
#                 domain_set=['restaurant','train']
#             exit()

        
        assert len(labels)==len(logs)
        labels=labels
        logs=logs
        print(len(self.new_labels),len(logs))
#         for index,data in enumerate(zip(labels,self.new_labels,logs)):
#             label,new_label,log=data[0],data[1],data[2]
#             if label['target']!=new_label['target']:
#                 print('index',index,log)
#                 break
#         else:
#             print('there is not error')
#         exit()
            
        database_dialog,database_label= load_database.load_database(self.extracted_data)
        print('database_dialog',len(database_dialog),'log', len([label for label in labels if label['target']==True]))
#         exit()
        if len(database_label)>len([label for label in labels if label['target']==True]):
            sampled_database=random.sample(list(zip(database_dialog,database_label)),len([label for label in labels if label['target']==True]))
        database_dialog=[ dialog_label_pair[0] for dialog_label_pair in sampled_database]
        database_label=[ dialog_label_pair[1] for dialog_label_pair in sampled_database]
        print('database_dialog',len(database_dialog),'log', len([label for label in labels if label['target']==True]))
#         exit()

        num=-1
        for log, label in tqdm(zip(logs, labels),disable=False):
#         for log, label,new_label in tqdm(zip(logs, labels,self.new_labels),disable=False):
            num+=1
            if label['target']:
                if (len(log) > 2 and self.utterance_window == 3):
                    premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
                              log[-2]['text'] + ". " + self.User_response_prefix + log[-1]['text']
                elif (len(log) > 1):
                    premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + log[-1]['text']
                else:
                    premise = log[-1]['text']

                # domain = label['knowledge'][0]['domain']
                knowledge_snippets=self.knowledge[label['knowledge'][0]['domain']][str(label['knowledge'][0]['entity_id'])]['docs'][str(label['knowledge'][0]['doc_id'])]['body']
                # positive_hypothesis = self.hypothesis_template.format(domain)
                positive_hypothesis=knowledge_snippets
                tokenized_positive_example = self.tokenizer(premise, positive_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                            , truncation=True)
                self.premise_hypothesis_examples.append(tokenized_positive_example)
                self.gt_labels.append(
                    torch.tensor([2]))
#                 self.knowledge_seeking_gts.append(True)
                entity_id=str(label['knowledge'][0]['entity_id'])
                domain=label['knowledge'][0]['domain']
                self.get_neg_database_knowledge_for_knowledge_example(premise,domain,entity_id)

                # torch.nn.NLLLoss requires only class idx
                # logits are in order [contradiction, neutral, entailment]
                remaining_knowledge=[knowledge_snippets['body'] for know_id,knowledge_snippets in self.knowledge[label['knowledge'][0]['domain']][str(label['knowledge'][0]['entity_id'])]['docs'].items() 
                                    if know_id is not str(label['knowledge'][0]['doc_id'])]
                # remaining_labels = [neg_domain for neg_domain in self.candidate_labels if neg_domain is not domain]
                

#                 print('num candidates',self.num_neg_candidates)
#                 print()
                if self.split_type=='train':
                    remaining_knowledge=random.sample(remaining_knowledge,min(self.num_neg_candidates,len(remaining_knowledge)))
                for neg_label in remaining_knowledge:
                    neg_know = neg_label
                    negative_hypothesis = neg_know
                    tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
                    self.premise_hypothesis_examples.append(tokenized_negative_example)
                    self.gt_labels.append(
                        torch.tensor([0]))  # logits are in order [contradiction, neutral, entailment]
# #                     self.knowledge_seeking_gts.append({'pred':True,'gt':True})
#             else:
#                 if num>=self.num_data:
#                     continue
#                 new_label=self.new_labels[num]
#                 if (len(log) > 2 and self.utterance_window == 3):
#                     premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
#                               log[-2]['text'] + ". " + self.User_response_prefix + log[-3]['text']
#                 elif (len(log) > 1):
#                     premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + log[-1]['text']
#                 else:
#                     premise = log[-1]['text']
# #                 print('index',num)
#                 assert label['target']==new_label['target']
# #                 print('new_label',new_label)
                
#                 domain=new_label['domain']
#                 name=new_label['name']
                
#                 positive_hypothesis=new_label['choice']
#                 tokenized_positive_example = self.tokenizer(premise, positive_hypothesis, return_tensors='pt',
#                                                         max_length=self.max_length, padding="max_length"
#                                                         , truncation=True)
#                 self.premise_hypothesis_examples.append(tokenized_positive_example)
#                 self.gt_labels.append(torch.tensor([2]))
#                 self.get_candidate_choices(premise,new_label)

        for log, label in tqdm(zip(database_dialog,database_label),disable=False):
            if (len(log) > 2 and self.utterance_window == 3):
                premise = self.User_response_prefix + log[-3]['text'] + ". " + self.System_response_prefix + \
                          log[-2]['text'] + ". " + self.User_response_prefix + log[-3]['text']
            elif (len(log) > 1):
#                 print('log',log)
                premise = self.System_response_prefix + log[-2]['text'] + ". " + self.User_response_prefix + log[-1]['text']
            else:
                premise = log[-1]['text']

            positive_hypothesis=label['info_sentence']

            tokenized_positive_example = self.tokenizer(premise, positive_hypothesis, return_tensors='pt',
                                                        max_length=self.max_length, padding="max_length"
                                                        , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_positive_example)
            self.gt_labels.append(torch.tensor([2]))
            self.get_neg_database_knowledge_for_database_example(premise,log,label)

    def get_neg_database_knowledge_for_knowledge_example(self,premise,domain,entity_id):
#         print(domain)
        with open(domain+'_db.json','r') as f:
            domain_database=json.load(f)
        candidates=[]
        if entity_id=='*':
            if isinstance(domain_database[0],dict):
                random_object=random.sample(domain_database,1)[0]
                for entity,value in random_object.items():
                    # candidates.append(self.hypothesis_template.format(domain,entity,value))
                    candidates.append(self.to_sentence({entity:value},domain,domain))

            else:
                for entity, value in domain_database.items():
                    candidates.append(self.to_sentence({entity:value},domain,domain))
        else:
            name=self.knowledge[domain][entity_id]['name']
#             print('name of knowledge is' ,name)
            if isinstance(domain_database[0],dict):
                name_entity=None
                for entity in domain_database[0].keys():
                    if entity in id_entity_name:
                        name_entity=entity
                        break
                else:
                    raise('There is a new name_entity, that not in id_entity_name')
                for object_detail in domain_database:
                    if object_detail[name_entity].lower()==name.lower():
                        
                        for entity, value in object_detail.items():
                            if entity!=name_entity:
                                candidates.append(self.to_sentence({entity:value},name,domain))
                        break
        if self.split_type=='train':
            candidates=random.sample(candidates,min(self.num_neg_candidates,len(candidates)))
#             candidates=random.sample(candidates,min(self.num_neg_candidates,len(candidates)))
        for negative_hypothesis in candidates:
#             print('candidate',candidates)
            tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_negative_example)
            self.gt_labels.append(torch.tensor([0]))
#             self.knowledge_seeking_gts.append({'pred':False,'gt':True})
#             exit()
    def get_candidate_choices(self,premise,new_label):
        domain=new_label['domain']
        name=new_label['name']
        gt_entity=new_label['correct_entity']
        with open(domain+'_db.json','r') as f:
            domain_database=json.load(f)
        remain_entities=[]
        neg_knowledge=[]
        name=None
        if isinstance(domain_database[0],dict):
            entities=self.extract_entities_by_name(name,domain_database,domain)
            if name==None:
                name=domain
            for entity,value in entities.items():
                if entity!=gt_entity and value!=name:
                    # candidate=self.hypothesis_template.format(name,entity,value)
                    candidate=self.to_sentence({entity:value},name,domain)
                    remain_entities.append(candidate)
        else:
            for entity, value in domain_database.items(): #the format of taxi_db need to be changed
                if entity!=gt_entity and value!=name:
                    # candidate=self.hypothesis_template.format(domain,entity,value)
                    candidate=self.to_sentence({entity:value},name,domain)
                    remain_entities.append(candidate)
                    
        if self.split_type=='train':
            remain_entities=random.sample(remain_entities,min(self.num_neg_candidates,len(remain_entities)))
        for negative_hypothesis in remain_entities:
            tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_negative_example)
            self.gt_labels.append(torch.tensor([0]))
#             self.knowledge_seeking_gts.append({'pred':False,'gt':False})

        # create negative example of knowledge snippets
        remain_entities=[]    
        
        for name_id, object_detail in self.knowledge[domain].items():
            if name_id!='*':
                if name!=None :
#                     print('name!=None',name,'|')
                
                    if object_detail['name'].lower()==name.lower():#may use other method to compare
#                         print(' in find the name',name)
                        for knowledge_snippet_id ,knowledge_snippet in object_detail['docs'].items():
                            candidate=knowledge_snippet['body']
                            remain_entities.append(candidate)
                else:
#                     print('name==None')
                    random_name_id=random.sample(list(self.knowledge[domain].keys()),1)[0]
                    
                    for knowledge_snippet_id ,knowledge_snippet in self.knowledge[domain][random_name_id]['docs'].items():
                        candidate=knowledge_snippet['body']
                        remain_entities.append(candidate)
                    break
            else:
#                 print('name_id==*')
                for knowledge_snippet_id ,knowledge_snippet in object_detail['docs'].items():
                    candidate=knowledge_snippet['body']
                    remain_entities.append(candidate)
        else:
            if remain_entities==[]:
                random_name_id=random.sample(list(self.knowledge[domain].keys()),1)[0]  
                for knowledge_snippet_id ,knowledge_snippet in self.knowledge[domain][random_name_id]['docs'].items():
                    candidate=knowledge_snippet['body']
                    remain_entities.append(candidate)
                
        if self.split_type=='train':
#             print('name of this example',name
#             print('len is ',len(remain_entities))
            remain_entities=random.sample(remain_entities,min(self.num_neg_candidates,len(remain_entities)))
        for negative_hypothesis in remain_entities:
            tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_negative_example)
            self.gt_labels.append(torch.tensor([0]))
    def get_neg_database_knowledge_for_database_example(self,premise,log,gt_label):
        domain=gt_label['domain']
        gt_entity=gt_label['info']
        with open(domain+'_db.json','r') as f:
            domain_database=json.load(f)
        cand_entNames=self.get_cand_entNames(domain_database)
        
        
        remain_entities=[]
        neg_knowledge=[]
        name=None
        if isinstance(domain_database[0],dict):
#             cand_entNames=self.get_cand_entNames(domain_database)
            if cand_entNames!=None:
#                 name=surfMatch.get_1ent(log,cand_entNames,known_domain = domain)
                name=gt_label['name']
                if name=='*'or name=='none' or name=='':
                    name=None
                elif name.lower()=='cafe uno':
                    name='caffe uno'
#                 elif name.lowe() in ['golden curry','cambridge belfry' ,'lensfield hotel','varsity restaurant','oak bistro','copper kettle','lucky star','nirala','cow pizza kitchen and bar','gardenia','gandhi','hotpot'] :
#                     name='the golden curry'
                elif name.lower()=='dojo noodle bar|j restaurant':
                    name='dojo noodle bar'
                elif name.lower()=='restaurant 22':
                    name='22 Chesterton Road Chesterton'
                elif name.lower()=='rosas bed and breakfast':
                    name='rosa\'s bed and breakfast'
                elif name.lower()=='pizza hut':
                    name='pizza hut city centre'
                name=self.fliter_name(name,domain)
#                 print('name',name)
                
#             if name=='t':
#                 print('name is t')
#                 print('log',log)
#                 exit()
#             print('log',log)
            entities=self.extract_entities_by_name(name,domain_database,domain)
            if name==None:
                name=domain
            for entity,value in entities.items():
                if entity!=gt_entity and value!=name:
                    # candidate=self.hypothesis_template.format(name,entity,value)
                    candidate=self.to_sentence({entity:value},name,domain)
                    remain_entities.append(candidate)
        else:
            for entity, value in domain_database.items(): #the format of taxi_db need to be changed
                if entity!=gt_entity and value!=name:
                    # candidate=self.hypothesis_template.format(domain,entity,value)
                    candidate=self.to_sentence({entity:value},name,domain)
                    remain_entities.append(candidate)
                    
        if self.split_type=='train':
            remain_entities=random.sample(remain_entities,min(self.num_neg_candidates,len(remain_entities)))
        for negative_hypothesis in remain_entities:
            tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_negative_example)
            self.gt_labels.append(torch.tensor([0]))
#             self.knowledge_seeking_gts.append({'pred':False,'gt':False})

        # create negative example of knowledge snippets
        remain_entities=[]    
        
        for name_id, object_detail in self.knowledge[domain].items():
            if name_id!='*':
                if name!=None :
#                     print('name!=None',name,'|')
                
                    if object_detail['name'].lower()==name.lower():#may use other method to compare
#                         print(' in find the name',name)
                        for knowledge_snippet_id ,knowledge_snippet in object_detail['docs'].items():
                            candidate=knowledge_snippet['body']
                            remain_entities.append(candidate)
                else:
#                     print('name==None')
                    random_name_id=random.sample(list(self.knowledge[domain].keys()),1)[0]
                    
                    for knowledge_snippet_id ,knowledge_snippet in self.knowledge[domain][random_name_id]['docs'].items():
                        candidate=knowledge_snippet['body']
                        remain_entities.append(candidate)
                    break
            else:
#                 print('name_id==*')
                for knowledge_snippet_id ,knowledge_snippet in object_detail['docs'].items():
                    candidate=knowledge_snippet['body']
                    remain_entities.append(candidate)
        else:
            if remain_entities==[]:
                random_name_id=random.sample(list(self.knowledge[domain].keys()),1)[0]  
                for knowledge_snippet_id ,knowledge_snippet in self.knowledge[domain][random_name_id]['docs'].items():
                    candidate=knowledge_snippet['body']
                    remain_entities.append(candidate)
                
        if self.split_type=='train':
#             print('name of this example',name
#             print('len is ',len(remain_entities))
            remain_entities=random.sample(remain_entities,min(self.num_neg_candidates,len(remain_entities)))
        for negative_hypothesis in remain_entities:
            tokenized_negative_example = self.tokenizer(premise, negative_hypothesis, return_tensors='pt',
                                                            max_length=self.max_length, padding="max_length"
                                                                , truncation=True)
            self.premise_hypothesis_examples.append(tokenized_negative_example)
            self.gt_labels.append(torch.tensor([0]))
#             self.knowledge_seeking_gts.append({'pred':True,'gt':False})
#         neg_knowledge=[neg_knowledge['body'] for idx,neg_knowledge in self.knowledge[domain][gt_entity]['dosc'].items()]

    def to_sentence(self,dic, name, dname):
        sentence = ''
        for k in dic.keys():
#             print('name',dic[k])
            value=dic[k]
            if type(value)!=str:
                if isinstance(value,dict):
                    value=list(value.keys())[0]
                if isinstance(value,list):
                    tmp_value=''
                    for item in value:
                        tmp_value+=str(item)+' '
                    value=tmp_value
                value=str(value)
#             if dic[k] == 'no':
#                 sentence += (name + ' has no ' + k + '. ')
#             elif dic[k] == 'yes':
#                 sentence += (name + ' has ' + k + '. ')
#             elif k == 'beginning':
#                 sentence += ('You are looking for the ' + name + ' and planning to book it.')
#             elif k == 'stars':
#                 sentence += (name + ' has stars of ' + value + '. ')
#             elif k == 'leaveAt':
#                 sentence += (name + ' leave at ' + value + '. ') 
#             elif k == 'arriveBy':
#                 sentence += (name + ' arrive by ' + value + '. ')
#             elif k == 'destination':
#                 sentence += (name + '\'s destination is ' + value + '. ')  
#             elif k == 'departure':
#                 sentence += (name + '\'s departure is ' + value + '. ') 
#             elif k == 'day':
#                 sentence += ('The day is ' + value + '. ') 
#             elif k == 'type' :
#                 sentence += ('The type is ' + dic[k] + '. ')
#             elif k == 'area' :
#                 sentence += ('The area of '+ name + ' is in the ' + value + '. ') 
#             elif k == 'name' :
#                 sentence += ('The name of this ' + dname + ' is '+ value +'. ') 
#             elif k=='ending':
# #                 print('it is ending in the neg candidate') 
#                 sentence += value
#             else:
#                 sentence += ('The '+ k + ' of ' + name + ' is ' + value + '. ')
            if dic[k] == 'no':
                sentence += (dname + ' has no ' + k + '. ')
            elif dic[k] == 'yes':
                sentence += (dname + ' has ' + k + '. ')
            elif k == 'beginning':
                sentence += ('You are looking for the ' + dname + ' and planning to book it.')
            elif k == 'stars':
                sentence += (dname + ' has stars of ' + value + '. ')
            elif k == 'leaveAt':
                sentence += (dname + ' leave at ' + value + '. ') 
            elif k == 'arriveBy':
                sentence += (dname + ' arrive by ' + value + '. ')
            elif k == 'destination':
                sentence += (dname + '\'s destination is ' + value + '. ')  
            elif k == 'departure':
                sentence += (dname + '\'s departure is ' + value + '. ') 
            elif k == 'day':
                sentence += ('The day is ' + value + '. ') 
            elif k == 'type' :
                sentence += ('The type is ' + dic[k] + '. ')
            elif k == 'area' :
                sentence += ('The area of '+ dname + ' is in the ' + value + '. ') 
            elif k == 'name' :
                sentence += ('The name of this ' + dname + ' is '+ value +'. ') 
            elif k=='ending':
#                 print('it is ending in the neg candidate') 
                sentence += value
            else:
                sentence += ('The '+ k + ' of ' + dname + ' is ' + value + '. ')
        if sentence=='The ending of hotel is You are welcome. Good bye..':
            print('entity name',k)
            exit()
        if sentence=='':
            print('the sentence is empty')
            exit()
        return sentence
    def extract_entities_by_name(self,name,domain_database,domain):
#         print('len of object',len(domain_database),'domain',domain)
        if name!=None:
            for object_detail in domain_database:
                for entity, value in object_detail.items():
                    if entity in id_entity_name:
#                         print('name',name,'entity name',value)
                        if value.lower()==name.lower():
#                             print('object',object_detail)
                            return object_detail
        
            else:
#                 print('name',name,domain)
#                 print()
#                 print('there is no such name')
#                 exit()
                return random.sample(domain_database,1)[0]
        else:
#             return domain_database[0]
            return random.sample(domain_database,1)[0]
    def fliter_name(self,name,domain):
#         print('name domain',self.entity_name_start_with_the[domain])
        if name==None:
            return None
        if '|' in name:
            name=name[:name.index('|')]
        if name in self.entity_name_start_with_the[domain]:
            return 'the '+name
        else:
            return name
    def preprocess_name(self):
        print('i am here')
        self.entity_name_start_with_the={}
        for domain in domain_set:
            self.entity_name_start_with_the[domain]=[]
            with open ( domain+'_db.json','r') as f:
                database_of_domain=json.load(f)
            entity_name=None
            if isinstance(database_of_domain[0],dict):
                for entity in database_of_domain[0].keys():
                    if entity in id_entity_name:
                        entity_name=entity
            else:
                continue
            if entity_name ==None:
                continue
            for object_detail in database_of_domain:
#                 print(object_detail[entity_name].lower())
                if object_detail[entity_name].lower().startswith('the '):
#                     print('the entity name',object_detail[entity_name].lower())
                    self.entity_name_start_with_the[domain].append(object_detail[entity_name].lower()[4:])
#         print('entity_name_start_with_the',self.entity_name_start_with_the.items())
#         exit()
    def get_cand_entNames(self,domain_database):
        entity_name=None
        cand_entNames=[]
        for entity in domain_database[0].keys():
            if entity in id_entity_name:
                entity_name=entity
        if entity_name==None:
            return None
        for object_detail in domain_database:
            cand_entNames.append(object_detail[entity_name])
        return cand_entNames
    def name_cls(log,domain):

        with open(domain+'_db.json','r') as f:
            domain_dataset=json.load(f)

        log_text=''
        log_type=0
        log_len=min(len(log),5)
        if log_len%2==1:
            log_type=0
        else:
            log_type=1
        for log_idx in log[-log_len:]:
            if log_type==0:
                log_text+=self.User_response_prefix+log['text']
                log_type=1
            else:
                log_text+=self.System_response_prefix+log['text']
                log_type=0

        name_set=[]
        if isinstance(domain_dataset[0],dict):
            for object_detail in domain_dataset:
                for entity, value in object_detail.items():
                    if entity in id_entity_name:
                        name_set.append(value)
                        break
                else:
                    raise('There is no name in this domain')

        ratios=[]
        for name in name_set:
            ratios=process.extract(name,log_text)
        # index= np.argmax(ratios)
        return name[np.argmax(ratios)]

    def __len__(self):
        return (len(self.gt_labels))

    def __getitem__(self, item):
        sample = self.premise_hypothesis_examples[item]
        label = self.gt_labels[item]
#         konwledge_seeking_gt=self.konwledge_seeking_gts[item]
        return sample, label
