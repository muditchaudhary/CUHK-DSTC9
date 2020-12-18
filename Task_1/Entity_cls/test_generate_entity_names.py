import json
import api_src.surfMatch as surfMatch
from tqdm import tqdm
entity_names_for_all_domains={}
domain_set=['attraction','hotel','restaurant','train','taxi']
id_entity_name=['name','trainID']
def generate_entity_names(log,domain):
	if domain=='train':
		return None
	if entity_names_for_all_domains[domain]==[]:
		return None
	# print('i am here')
	name=surfMatch.get_1ent(log,entity_names_for_all_domains[domain],known_domain = domain)
	return name
def get_names_for_all_domains():
    for domain in domain_set:
        # entity_names_for_all_domains[domain]=[]
        # with open(domain+'_db.json') as f:
        #     domain_database=json.load(f)
        # entity_name=None
        # for entity in domain_database[0].keys():
        #     if entity in id_entity_name:
        #         entity_name=entity
        # if entity_name==None:
        #     continue
        # for object_detail in domain_database:
        #     entity_names_for_all_domains[domain].append(object_detail[entity_name])
        entity_names_for_all_domains[domain]=[]
        with open('test/knowledge.json','r') as f:
        	knowledge=json.load(f)
        for entity,value in knowledge[domain].items():
        	if value['name'] not in entity_names_for_all_domains and value['name']!=None:
	        	entity_names_for_all_domains[domain].append(value['name'])

def main():
	with open('test/logs.json','r') as f:
		logs=json.load(f)
	with open('test_domain_cls_split8.json','r') as f:
		domain_preds=json.load(f)
	preds=[]
	print(len(logs),len(domain_preds))
	assert len(logs)==len(domain_preds)
	get_names_for_all_domains()
	# print(entity_names_for_all_domains)
	# exit()
	num=0
	for log , domain_pred in tqdm(zip(logs,domain_preds)):
		name_pred={}
		name_pred['name']=generate_entity_names(log,domain_pred['Pred_domain'])
		name_pred['log']=log
		name_pred['domain']=domain_pred['Pred_domain']
		preds.append(name_pred)
		# num+=1
		# if num==10:
		# 	break
	with open('test_name_cls_split8.json','w') as f:
		json.dump(preds,f,indent=4)
main()