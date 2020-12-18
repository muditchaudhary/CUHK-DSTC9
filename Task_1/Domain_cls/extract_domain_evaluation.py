#from fuzzywuzzy import process
import json
import numpy as np
from transformers import pipeline
classifier = pipeline("zero-shot-classification",model="./runs_POD_version2/checkpoint-128")
from tqdm import tqdm
def extract_domain(log_file,label_file,know_cands):
    with open(log_file,'r') as f:
        logs=json.load(f)
    with open(label_file,'r') as f:
        labels=json.load(f)
    with open(know_cands,'r') as f :
        know_cands=json.load(f)
    tn,tp,fn,fp=0,0,0,0
    assert len(logs)==len(labels)
    id=0
    tp_log=[]
    fp_log=[]
    hypothesis = "The user is asking about {}."
    for log, label in tqdm(zip(logs,labels)):        
        detected_domain=None
        # print(log,label)
        if label['target']==True:
            if(len(log)>2):
                turn_text = "Assistant says " + log[-2]['text'] + ". User says " + log[-1]['text']
            elif (len(log)==2):
                turn_text = log[-2]['text']+" "+ log[-1]['text']
            else:
                turn_text=log[-1]['text']
            preds=classifier(turn_text,['restaurant','taxi'], hypothesis_template=hypothesis)
            pred_label = preds['labels'][0]
            pred_score = preds['scores']

            max_index=np.argmax(pred_score)

            detected_domain = pred_label

            label_domain=label['knowledge'][0]['domain']
            if label_domain==detected_domain:
                tp+=1
                tp_log.append([turn_text,label['knowledge'],detected_domain,preds])
            else:
                fp_log.append([turn_text,label['knowledge'],detected_domain,preds])
                fp+=1
            id+=1
    with open('Evaluations/fp_summary_val_run_POD_domain_version2_128.json', 'w') as f:
        json.dump(fp_log,f,indent=4)
    with open('Evaluations/tp_summary_val_run_POD_domain_version2_128.json', 'w') as f:
        json.dump(tp_log,f,indent=4)
    print('accuarcy: ',tp/(tp+fp))

extract_domain('./POD_domain/version2/val/logs.json','./POD_domain/version2/val/labels.json','knowledge.json')
