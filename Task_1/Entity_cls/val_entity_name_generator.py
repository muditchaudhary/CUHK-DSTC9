import json
import api_src.surfMatch as surfMatch
from tqdm import tqdm
entity_names_for_all_domains={}
domain_set=['hotel','restaurant','train','taxi']
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
        entity_names_for_all_domains[domain]=[]
        with open('old/'+domain+'_db.json') as f:
            domain_database=json.load(f)
        entity_name=None
        for entity in domain_database[0].keys():
            if entity in id_entity_name:
                entity_name=entity
        if entity_name==None:
            continue
        for object_detail in domain_database:
            entity_names_for_all_domains[domain].append(object_detail[entity_name])
def main():
	with open('val_mudit/logs.json','r') as f:
		logs=json.load(f)
	with open('domain_cls_val_split_8_2.json','r') as f:
		domain_preds=json.load(f)
	with open('val_mudit/labels.json','r') as f:
		labels=json.load(f)
	with open('val_mudit/knowledge.json','r') as f:
		knowledges=json.load(f)
	preds=[]

	print(len(logs),len(domain_preds))
	assert len(logs)==len(domain_preds)
	get_names_for_all_domains()
	# print(entity_names_for_all_domains)
	# exit()
	num=0
	for log , domain_pred,label in tqdm(zip(logs[:39905],domain_preds[:39905],labels[:39905])):
		if label['target']==False:

			name_pred={}
			name_pred['name']=generate_entity_names(log,domain_pred['Pred_domain'])
			name_pred['log']=log
			name_pred['domain']=domain_pred['Pred_domain']
			name_pred['target']=False
			preds.append(name_pred)
		
		else:
			name_pred={}
			domain=label['knowledge'][0]['domain']
			entity_id=label['knowledge'][0]['entity_id']
			# print(label['knowledge'][0]['domain'],entity_id)
			# exit()
			name_pred['name']=knowledges[domain][str(entity_id)]['name']
			name_pred['log']=log
			name_pred['domain']=domain
			name_pred['target']=True
			preds.append(name_pred)
		# num+=1
		# if num==10:
		# 	break
		# break

	with open('mudit.json','w') as f:
		json.dump(preds,f,indent=4)
main()