# from fuzzywuzzy import process
import json
import numpy as np
from transformers import pipeline
import argparse

from tqdm import tqdm


def extract_domain(log_file, out_file, labels, classifier):
    predictions=[]
    with open(log_file, 'r') as f:
        logs = json.load(f)
        f.close()

    print("The length of logs is {}".format(str(len(logs))))
    hypothesis = "The user is asking about {}."
    for i, log in tqdm(enumerate(logs)):

        if (len(log) >= 2):
            turn_text = "Assistant says " + log[-2]['text'] + ". User says " + log[-1]['text']
        else:
            turn_text = "User says " + log[-1]['text']
        preds = classifier(turn_text, labels, hypothesis_template=hypothesis)
        pred_label = preds['labels'][0]
        log = {"Preds":preds, "Pred_domain":pred_label}

        predictions.append(log)

    with open(out_file, 'w') as f:
        json.dump(predictions, f, indent=4)
        f.close()

    with open(out_file) as f:
        predictions_check = json.load(f)

    assert len(logs) == len(predictions_check), "predictions-log length mismatch"
    print("Predictions saved in {}".format(out_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="File for inference")
    parser.add_argument("--out_file", help="File to store inferences")
    parser.add_argument("--model_path", help="model or path")
    parser.add_argument('--labels', '--list', nargs='+', help='Labels', required=True)

    args = parser.parse_args()
    print("The data_file is {}".format(args.data_file))
    print("The out_file is {}".format(args.out_file))
    print("The labels are {}".format(args.labels))
    classifier = pipeline("zero-shot-classification", model=args.model_path)

    extract_domain(args.data_file, args.out_file, args.labels, classifier=classifier)