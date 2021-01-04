import json
from tqdm import tqdm
from transformers import pipeline

def extract_domain(premise, labels_list,model_path="./models//domain_cls_model"):
    classifier = pipeline("zero-shot-classification", model=model_path)
    hypothesis = "The user is asking about {}."
    preds = classifier(premise, labels_list, hypothesis_template=hypothesis)
    pred_label = preds['labels'][0]
    return {"Preds":preds, "Pred_domain":pred_label}