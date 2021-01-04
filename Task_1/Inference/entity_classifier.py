import json
import api_src.surfMatch as surfMatch
from tqdm import tqdm

entity_names_for_all_domains = {}
domain_set = ['attraction', 'hotel', 'restaurant', 'train', 'taxi']
id_entity_name = ['name', 'trainID']


def extract_entity(log, domain, entity_list):
    if domain == 'train':
        return None
    if entity_list[domain] == []:
        return None
    name = surfMatch.get_1ent(log, entity_list[domain], known_domain=domain)
    return name
