from openai import OpenAI
import os
import tqdm
import time
import sys
#from nli_model import NLIModel
from collections import Counter
import json
import argparse
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')

def evaluate_summaries(prediction, gold):
    # Calculate METEOR
    meteor = meteor_score([word_tokenize(gold)], word_tokenize(prediction))
    
    # Calculate ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, gold)
    
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    token_count = len(word_tokenize(prediction))
    
    return {
        "METEOR": meteor,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "token_length": token_count,
    }

def process_files(predictions_folder, dataset_file, data_folder):
    with open(dataset_file, 'r') as f:
        dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()]

    global_eval_results = {
        "METEOR": 0,
        "ROUGE-1": 0,
        "ROUGE-2": 0,
        "ROUGE-L": 0,
        "token_length": 0,
    }

    # Iterate through each text file in the folder
    for filename in tqdm.tqdm(dataset_lines):
        if True:
            prediction_file_path = os.path.join(predictions_folder, filename)
            gold_file_path = os.path.join(data_folder, filename)

            # Read content from the text file
            with open(prediction_file_path, 'r') as file:
                prediction = file.read()
            
            # Read content from the text file
            with open(gold_file_path, 'r') as file:
                gold = file.read()

            eval_results = evaluate_summaries(prediction, gold)

            for key in global_eval_results:
                global_eval_results[key] += eval_results[key]

    for key in global_eval_results:
        global_eval_results[key] /= len(dataset_lines)

    print(global_eval_results)

# Example usage
dataset_file = "dataset/test.tsv"
predictions_folder = 'data_2'
data_folder = "data_2"
process_files(predictions_folder, dataset_file, data_folder)