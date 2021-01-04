try:
    __IPYTHON__
    USING_IPYTHON = True
except NameError:
    USING_IPYTHON = False

from collections import defaultdict, OrderedDict
import copy
import re
import json
import argparse
from pprint import pprint
import string
try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm    
    
from fuzzywuzzy import fuzz
from Levenshtein import distance, hamming
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
import numpy as np

from orderedset import OrderedSet



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(lmtzr ,word, pos = wordnet.NOUN, lmtz = True):
    if not pos:
        pos = wordnet.NOUN
    if lmtz and lmtzr != None:
        return lmtzr(word, pos = pos)


def gen_aliases(name, domain, do_alias = True):
    
#     prefixes = ["RESTAURANT", "LODGE", "HOTEL", "GUESTHOUSE", "HOUSE", "CAFE", "BAR"]
#     suffixes = ["RESTAURANT", "LODGE", "HOTEL", "GUESTHOUSE", "HOUSE", "CAFE", "BAR"]
    prefixes = [domain.upper()]
    suffixes = [domain.upper()]
    synonyms_sets = [["&", " & ", "AND", " AND "]]
    oneway_synonyms_sets = {"-": ["", " "], "(^LA | LA | LA$)": [" THE "]}
    if do_alias:
        prefixes = list(set(prefixes + ["RESTAURANT", "HOTEL", "GUESTHOUSE", "HOUSE", "CAFE", "BAR", "THE"]))
        suffixes = list(set(suffixes + ["RESTAURANT", "HOTEL", "GUESTHOUSE", "HOUSE", "CAFE", "BAR"]))
        synonyms_sets.append(["GUESTHOUSE", "GUEST HOUSE"])
    
    aliases = OrderedSet()
    aliases_tmp = aliases.copy()
    
    name_upper = name.upper()
    aliases_tmp.add(name_upper)

#     print (aliases)
#     print ()

    
    # generate alias by replacing synonyms with each other
    while(len(aliases_tmp) != len(aliases)):
        aliases = aliases_tmp.copy()
        for alias in aliases:
            for synonyms in synonyms_sets:
                for idx in range(len(synonyms)):
                    replacing_syn = synonyms[idx]
    #                 print (replacing_syn)
                    matching_syn = synonyms.copy()
    #                 matching_syn.pop(idx)
    #                 print (matching_syn)
                    rmSuffix_name = re.sub(("|").join(matching_syn), replacing_syn, alias)
                    rmSuffix_name_strip = re.sub(" +", " ", rmSuffix_name).strip()
                    aliases_tmp.add(rmSuffix_name_strip)

#     pprint (aliases_tmp)
#     print ()

    # generate for one-way synonym replaced aliases
    aliases = OrderedSet()
    while(len(aliases_tmp) != len(aliases)):
        aliases = aliases_tmp.copy()
        for alias in aliases:
            for matching_syn in oneway_synonyms_sets:
                for replacing_syn in oneway_synonyms_sets[matching_syn]:
                    rmSuffix_name = re.sub(matching_syn, replacing_syn, alias)
                    rmSuffix_name_strip = re.sub(" +", " ", rmSuffix_name).strip()
                    aliases_tmp.add(rmSuffix_name_strip)
                    
    # generate alias by removing suffixes
    aliases = OrderedSet()
    while(len(aliases_tmp) != len(aliases)):
        aliases = aliases_tmp.copy()
        for alias in aliases:
            for suffix in suffixes:
                rmSuffix_name = re.sub("(.*) " + suffix + "$", "\g<1>", alias)
                rmSuffix_name_strip = re.sub(" +", " ", rmSuffix_name).strip()
                aliases_tmp.add(rmSuffix_name_strip)
                
                        
    # generate alias by removing prefixes
    aliases = OrderedSet()
    while(len(aliases_tmp) != len(aliases)):
        aliases = aliases_tmp.copy()
        for alias in aliases:
            for prefix in prefixes:
                rmPrefix_name = re.sub("^" + prefix + " (.*)", "\g<1>", alias)
                rmPrefix_name_strip = re.sub(" +", " ", rmPrefix_name).strip()
                aliases_tmp.add(rmPrefix_name_strip)
                
    return aliases_tmp



def matched_name_in_snt(name, alias_names, snt, mode):
    user_utterance = snt
    domEnt = name
    alias_domEnts = alias_names
    # check 4968
    if mode == 'exact':
        # hard matching
        searchObj = re.search("(^{}\W|\W{}\W|\W{}$)".format(domEnt, domEnt, domEnt), user_utterance, re.I)
        if searchObj:
            return True
        
    if mode == 'hamming':
        # hamming distance
        if len(domEnt) <= 4:
            return False
        
#         if not "acorn" in domEnt.lower():
#             return False
        
        for start in range(len(user_utterance) - len(domEnt) + 2):
            ## if not segmented properly, continue 
            
#             if "acorn" in domEnt.lower():
#                 print (user_utterance.lower()[start: start + len(domEnt) - 1])
#                 print (start, len(domEnt), len(user_utterance))
            if (
                    start == 0 and start + len(domEnt) < len(user_utterance) and\
                        not user_utterance[start + len(domEnt)].isalnum() or\
                    start == len(user_utterance) - len(domEnt) and start > 0 and\
                        not user_utterance[start-1].isalnum() or\
                    (start-1 > 0 and not user_utterance[start-1].isalnum() and\
                     start + len(domEnt) < len(user_utterance) and\
                     not user_utterance[start + len(domEnt)].isalnum())
                ):
                

                if hamming(domEnt.lower(), user_utterance.lower()[start: start + len(domEnt)]) <= 1 and user_utterance[start].lower() == domEnt[0].lower():# and\
                   #user_utterance[start].isupper():
                    # print ("hamming0: ", domEnt, user_utterance[start: start + len(domEnt)])
                    return True
            # +1
            if (
                    start == 0 and start + len(domEnt) + 1 < len(user_utterance) and\
                        not user_utterance[start + len(domEnt) + 1].isalnum() or\
                    start == len(user_utterance) - len(domEnt) - 1 and start > 0 and\
                        not user_utterance[start-1].isalnum() or\
                    (start-1 > 0 and not user_utterance[start-1].isalnum() and\
                     start + len(domEnt) + 1 < len(user_utterance) and\
                     not user_utterance[start + len(domEnt) + 1].isalnum())
                ):
                
                if fuzz.ratio(domEnt.lower(), user_utterance.lower()[start: start + len(domEnt) + 1]) >= 90 and user_utterance[start].lower() == domEnt[0].lower():# and\
                   #user_utterance[start].isupper():
        #                         print ("hamming: ", domEnt, user_utterance[start: start + len(domEnt)])
#                     print ("hamming+1: ", domEnt, user_utterance)
                    return True
    
            # -1
            if (
                    start == 0 and start + len(domEnt) - 1 < len(user_utterance) and\
                        not user_utterance[start + len(domEnt) - 1].isalnum() or\
                    start == len(user_utterance) - len(domEnt) + 1 and start > 0 and\
                        not user_utterance[start-1].isalnum() or\
                    (start-1 > 0 and not user_utterance[start-1].isalnum() and\
                     start + len(domEnt) - 1 < len(user_utterance) and\
                     not user_utterance[start + len(domEnt) - 1].isalnum())
                ):
#                 print (user_utterance[start: start + len(domEnt) - 1])
                    
                
                
                if fuzz.ratio(domEnt.lower(), user_utterance.lower()[start: start + len(domEnt) - 1]) >= 90 and user_utterance[start].lower() == domEnt[0].lower():# and\
                   #user_utterance[start].isupper():
#                     print ("hamming-1: ", domEnt, user_utterance.lower()[start: start + len(domEnt) - 1])
                    return True
                      
    
    if mode == 'alias':
        # alias matching
        for domEnt in alias_domEnts:
        #                     print ("try alias: ", domEnt, user_utterance)
            searchObj = re.search("(^{}\W|\W{}\W|\W{}$)".format(domEnt, domEnt, domEnt), user_utterance, re.I)
            if searchObj:
                if user_utterance[searchObj.span(1)[0]+1].isupper():
                    return True
                else:
                    # print (domEnt, "vs", user_utterance)
                    return False
            # if upper then cannot match 5325, 5326, etc, where user types lowercase incomplete entity name
            # but if not upper, then ask will be identified as Ask Restaurant 
            
#             for start in range(len(user_utterance) - len(domEnt) + 2):
#                 ## if not segmented properly, continue 

#     #             if "acorn" in domEnt.lower():
#     #                 print (user_utterance.lower()[start: start + len(domEnt) - 1])
#     #                 print (start, len(domEnt), len(user_utterance))
#                 if (
#                     start == 0 and start + len(domEnt) < len(user_utterance) and\
#                         not user_utterance[start + len(domEnt)].isalnum() or\
#                     start == len(user_utterance) - len(domEnt) and start > 0 and\
#                         not user_utterance[start-1].isalnum() or\
#                     (start-1 > 0 and not user_utterance[start-1].isalnum() and\
#                      start + len(domEnt) < len(user_utterance) and\
#                      not user_utterance[start + len(domEnt)].isalnum())
#                     ):


#                     if hamming(domEnt.lower(), user_utterance.lower()[start: start + len(domEnt)]) <= 1 and user_utterance[start].lower() == domEnt[0].lower():# and\
#                        #user_utterance[start].isupper():
#                         print ("alias hamming: ", domEnt, user_utterance)
#                         return True
      
    if mode == 'lemma':
        lemmatizer = WordNetLemmatizer().lemmatize # lemmatizer function or None
        tokenizer = nltk.tokenize.WordPunctTokenizer().tokenize
        
        tokenized = tokenizer(user_utterance)
        tokenized_pos = nltk.pos_tag(tokenized)
        tokenized_joined = " ".join([lemmatize(lemmatizer, token.lower(),
                                               get_wordnet_pos(pos) or wordnet.NOUN).lower()
                                     for token, pos in tokenized_pos])
        # lemma matching
        searchObj = re.search("(^{}\W|\W{}\W|\W{}$)".format(domEnt, domEnt, domEnt), tokenized_joined, re.I)
        if searchObj:
            return True
    return False




# keep an entity for one domain, if matched
def get_1ent(log, cand_entNames, do_alias = True, known_domain = ""):
    
    # reverse log
    rev_log = copy.deepcopy(log)
    rev_log.reverse()
    
    domainsAndEntities_name_tomatch = cand_entNames
    domainsAndEntities_name2alias = defaultdict(set)
    
    # augment the names to make matching more robust
    for name in domainsAndEntities_name_tomatch:
        domainsAndEntities_name2alias[name] = gen_aliases(name, known_domain, do_alias)
    
    ord_targDom2ent_id = OrderedDict()
    
    ### Begin matching domain and entities ###
    target_entity_id = None
    target_domain = None
    target_domains = set()
    
    target_domains.add(known_domain)
        
    for turn in rev_log:
        user_utterance = turn['text'] 

        # exact
        for domEnt in domainsAndEntities_name_tomatch:
            exact_matched = matched_name_in_snt(domEnt,
                                                domainsAndEntities_name2alias[domEnt],
                                                user_utterance, mode = "exact")
            if exact_matched:
    #                     print ("exact: ", domEnt, user_utterance)
                return domEnt

        # hamming
        for domEnt in domainsAndEntities_name_tomatch:
            hamming_matched = matched_name_in_snt(domEnt,
                                                  domainsAndEntities_name2alias[domEnt],
                                                  user_utterance, mode = "hamming")
            if hamming_matched:
    #                     print ("hamming: ", domEnt, user_utterance)
                return domEnt

        # alias
        for domEnt in domainsAndEntities_name_tomatch:
            alias_matched = matched_name_in_snt(domEnt,
                                                domainsAndEntities_name2alias[domEnt],
                                                user_utterance, mode = "alias")
            if alias_matched:
    #                     print ("alias: ", domEnt, user_utterance)                 
                return domEnt

        # lemma
        for domEnt in domainsAndEntities_name_tomatch:
            lemma_matched = matched_name_in_snt(domEnt,
                                                domainsAndEntities_name2alias[domEnt],
                                                user_utterance, mode = "lemma")

            if lemma_matched:
    #                     print ("lemma: ", domEnt, user_utterance)              
                return domEnt
            
    # domain and entity not matched after all strategies
    # Use bert top 1
    # or return None
    return None
