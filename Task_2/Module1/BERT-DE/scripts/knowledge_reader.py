import os
import json

class KnowledgeReader(object):
    def __init__(self, dataroot, knowledge_file):
        path = os.path.join(os.path.abspath(dataroot))

        with open(os.path.join(path, knowledge_file), 'r') as f:
            self.knowledge = json.load(f)

#     * domain\_id: domain identifier (string: "hotel", "restaurant", "train", "taxi", etc.)
#       * entity\_id: entity identifier (integer or string: "*" for domain-wide knowledge)
#           * name: entity name (string; only exists for entity-specific knowledge)

    
    
    # structure:
    #       {domain_id: {entity_id: {name: xxxx}}}
    
    # return all domains (hotel, restaurant, taxi, train)
    # list of strings (length = 4)
    def get_domain_list(self):
        return list(self.knowledge.keys())

    # given domain ==> return all entities 'id', 'name' (of entities) within this domain
    # list of dictionaries:
    #                       [{entity_id: xxx, entity_name: xxx},
    #                        {},
    #                        ...]
    def get_entity_list(self, domain):
        # invalid domain
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name")

        entity_ids = []
        # all entities in this domain
        for entity_id in self.knowledge[domain].keys():
            try:
                entity_id = int(entity_id)
                entity_ids.append(int(entity_id))
            except:
                pass
        
        # entity_id, entity_name
        result = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            result.append({'id': entity_id, 'name': entity_name})

        return result

    # given domain & entity_id ==> return entity name
    # string of entity name
    def get_entity_name(self, domain, entity_id):
        # invalid domain
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)
        # invalid entity_id
        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = self.knowledge[domain][str(entity_id)]['name'] or None

        return result
    
    def get_domain_entity_list(self, domain=None, entity_id=None):
        if domain is None:
            domain_list = self.get_domain_list()
        else:
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name: %s" % domain)
            domain_list = [domain]
        
        result = []
        for domain in domain_list:
            if entity_id is None:
                for item_id, item_obj in self.knowledge[domain].items():
                    item_name = self.get_entity_name(domain, item_id)
                    if item_id != '*':
                        item_id = int(item_id)
                    if item_obj['name'] != item_name: print("Entity name not match!\t", item_obj['name'], '\t',item_name)
                    result.append({'domain': domain, 'entity_id': item_id, 'entity_name': item_name})
            else:
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id: %s" % str(entity_id))
                entity_name = self.get_entity_name(domain, entity_id)
                entity_obj = self.knowledge[domain][str(entity_id)]
                result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name})
        return result



    # given domain (opt), entity_id (opt) ==> return all docs (of this entity / domain)
    # list of dictionaries:
    #                       [{domain: xxx, entity_id: xxx, entity_name: xxx, doc_id: xxx, doc: {title: xxx, body: xxx}}
    #                        {}
    #                       ...]
    def get_doc_list(self, domain=None, entity_id=None):
        # domain not given ==> get domain list (4 possible strings)
        if domain is None:
            domain_list = self.get_domain_list()
        # domain given
        else:
            # invalid domain
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name: %s" % domain)
            domain_list = [domain]

        result = []
        for domain in domain_list:
            # entity_id not given
            if entity_id is None:
                # iterate over entities
                for item_id, item_obj in self.knowledge[domain].items():
                    # get entity name
                    item_name = self.get_entity_name(domain, item_id)
                    # has entity (hotel, restaurant)
                    if item_id != '*':
                        item_id = int(item_id)
                    # for one entity, iterate all docs
                    for doc_id, doc_obj in item_obj['docs'].items():
                        result.append({'domain': domain, 'entity_id': item_id, 'entity_name': item_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
            # entity_id given
            else:
                # invalid entity_id
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id: %s" % str(entity_id))

                entity_name = self.get_entity_name(domain, entity_id)
                
                entity_obj = self.knowledge[domain][str(entity_id)]
                # for one entity, iterate all docs
                for doc_id, doc_obj in entity_obj['docs'].items():
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
        return result

    # given domain, entity_id, doc_id ==> return doc
    # dictionary: {domain: xxx, entity_id: xxx, entity_name: xxx, doc_id: xxx, doc: {title: xxx, body: xxx}}
    def get_doc(self, domain, entity_id, doc_id):
        # invalid domain
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)
        # invalid entity_id
        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)
        # invalid doc_id
        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['docs']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['docs'][str(doc_id)]
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}}

        return result
