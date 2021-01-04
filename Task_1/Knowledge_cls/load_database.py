def load_database(database):
    database_dialogs=[]
    database_labels=[]
    for data in database:
        database_dialogs.append(data['utterance'])
        database_labels.append({'domain':data['domain'],'name':data['name'],'info_sentence':data['info_sentence'], 'info': data['info']})
    return database_dialogs,database_labels
                    
