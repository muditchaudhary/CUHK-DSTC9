
import json
import copy
from scripts.knowledge_reader import KnowledgeReader
import argparse

#read knowledge in the knowledge json file into a dictionary
def read_knowledge():
    knowledge_reader = KnowledgeReader(args.kw_path,'knowledge.json' )
    knowledge = knowledge_reader.knowledge
    knowledge_docs = knowledge_reader.get_doc_list()
    snippets = dict()
    for snippet in knowledge_docs:
        key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
        knowledge = snippet["doc"]["body"]
        snippets[key] = knowledge
    return snippets

#map a retrieved snippet to knowledge text
def get_kw_text(snippet, snippets):
    key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
    knowledge = snippets[key]
    return knowledge

#split the response into two part by "."
def split_by_dot(utt):
    utt_split = utt.strip('').split('.')
    utt_list = [u for u in utt_split if u != '']
    if len(utt_list) < 2 :  
        str1,str2 = split_by_upper(utt)
    elif len(utt_list) == 2:
        str2 = utt_list[1][1:]
        str1 = utt_list[0]+'.'
    else:
        str2 = utt_list[-1][1:]
        str1 = '.'.join(utt_list[:-1])+'.'        
    return str1, str2

# if the dot fails, try use uppercase to identify two components. 
def split_by_upper(utt):
    word_list = utt.split(' ')
    upper_index = []
    for i,w in enumerate(word_list):
        if w not in  ['I',''] and w[0].isupper():
            upper_index.append(i)
    if len(upper_index) < 2:
        str1 = utt
        str2 = utt
    else:
        start_token = upper_index[-1]
        str1 = ' '.join(word_list[:start_token])+'.'
        str2 = ' '.join(word_list[start_token:])
    return str1,str2

def replace(snippets):
    gen_res_dict = {}  
    best  = json.load(open(args.input_file))
    knowledge_replaced_task2 = copy.deepcopy(best)
    for i in range(len(best)):  
        if best[i]['target']:
            response = best[i]['response']
            task2_knowledge = get_kw_text(best[i]['knowledge'][0], snippets)
            _,response_2 = split_by_dot(response)
            concate_2 = task2_knowledge +' '+ response_2
            knowledge_replaced_task2[i]['response'] = concate_2
    output_file = args.input_file[:-5]+'_recon.json'
    replace_task2_str = json.dumps(knowledge_replaced_task2,indent=4)
    with open(output_file,'w') as f:
        f.write(replace_task2_str)
    print('writing RR result to %s.' % output_file)
    return knowledge_replaced_task2

def ensemble(snippets):
    in_domain = ['hotel','taxi','restaurant']
    gen  = json.load(open(args.input_file))
    rr = replace(snippets)
    probs = json.load(open(args.task2_probs))

    if len(probs) != len(gen):
        print('Please check if your probs file go with the task 2 output file')
        return 0
    ens = copy.deepcopy(gen)
    count = 0
    total = 0
    for i in range(len(gen)):
        if gen[i]['target']:
            domain = gen[i]['knowledge'][0]['domain']
            if domain not in in_domain:
                p = probs[i]
                if p[1] != 0 or p[0]/p[1] >= 5:
                    ens[i] = rr[i]
                    count += 1
            total += 1
    ens_str = json.dumps(ens,indent=4)
    output_file = args.input_file[:-5]+'_rr.json'
    with open(output_file,'w') as f:
        f.write(ens_str)
    print('writing Ens-GPT result to %s.' % output_file)
    print('%f responses were replaced by RR output.' % (count/total))








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kw_path", help="path of the knowledge.json", type=str, default="./data/")
    parser.add_argument('--input_file', help="output of task 3", type=str)
    parser.add_argument('--task2_probs', help="output probablities of task 2", type=str)
    parser.add_argument('--mode', help="RR or Ens", type=str)
    print('working')
    args = parser.parse_args()
    snippets = read_knowledge()
    if args.mode == 'RR':
        replace(snippets)
    else:
        ensemble(snippets)
