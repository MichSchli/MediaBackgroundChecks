from openai import OpenAI
import os
import tqdm
import time
import sys
#from nli_model import NLIModel
from collections import Counter
import json
import argparse

def most_frequent_element(input_list):
    return Counter(input_list).most_common(1)[0][0] if input_list else None

# Read the config file to get the openai key:
with open("config.json", "r") as f:
    config = json.load(f)
    openai_key = config["openai_key"]

client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_key
)

#nli_checker = NLIModel(model_name='cross-encoder/nli-deberta-v3-large')

def send_message(message):
    model = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=message
    )
    text = response.choices[0].message.content

    return text

def check_implication_deberta(claim, text):
    entails = nli_checker.check_text_implication(text, claim)
    return entails

def check_implication(claim, text, passes=4):
    votes = [gpt_check_implication(claim, text) for _ in range(passes)]

    majority = most_frequent_element(votes)

    return majority

def gpt_check_implication(question, text):
    # TODO replace with an NLI model
    system_message = "You are FactCheckGPT, a world-class tool used by journalists to discover problems in their writings. Users give you text, and check whether facts are true given the text. You ALWAYS answer either TRUE, FALSE, or NOT ENOUGH EVIDENCE."
    prompt = "You will be given a snippet written as part of a source criticism exercise, and a claim. Your task is to determine whether the claim is true based ONLY on the text. Do NOT use any other knowledge source\n\n"
    prompt += "The claim is: \"" + question + "\".\n"
    prompt += "The text follows below:\n\"" + text + "\".\n\n"
    prompt += question + " Thinking step by step, answer either TRUE, FALSE, or NOT ENOUGH EVIDENCE, capitalizing all letters. Explain your reasoning FIRST, and after that output either TRUE, FALSE, or NOT ENOUGH EVIDENCE."

    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    
    retries = 0
    result = None
    while result is None:
        try:
            result = send_message(messages)
        except:
            print(f"Failed to send message. Retrying (attempt {retries + 1})...", file=sys.stderr)
            delay = min(1 * (2 ** retries), 64)
            time.sleep(delay)
            retries += 1

    if "TRUE" in result:
        return "entailment"
    elif "FALSE" in result:
        return "contradiction"
    else:
        return "neutral"

def generate_atomic_fact(question, text):
    system_message = "You are InfoHuntGPT, a world-class tool used by journalists to quickly extract claims from text."
    prompt = "You will be given a snippet written as part of a source criticism exercise, and a fill-in-the-blank question (blanks represented by _). Your task is to fill in the blanks in the sentence, adding no additional information or wording. JUST replace the _ character. No yapping.\n\n"
    prompt += "The question is:\n" + question + "\n\n"
    prompt += "The text follows below:\n\"" + text + "\".\n\n"
    prompt += "Fill in the blanks in the question, adding no additional information or wording. JUST replace the _ character, and output ONLY the question with the blank filled in. No yapping."

    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    
    retries = 0
    result = None
    while result is None:
        try:
            result = send_message(messages)
        except:
            print(f"Failed to send message. Retrying (attempt {retries + 1})...", file=sys.stderr)
            delay = min(1 * (2 ** retries), 64)
            time.sleep(delay)
            retries += 1

    return result

def process_files(predictions_folder, dataset_file, fact_folder, start_at=0, end_at=-1):
    with open(dataset_file, 'r') as f:
        dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()][start_at:end_at]

    recalled_info_rate = 0
    error_rate = 0

    # Iterate through each text file in the folder
    for filename in tqdm.tqdm(dataset_lines):
        this_file_info_rate = 0
        this_file_error_rate = 0
        if True:
            prediction_file_path = os.path.join(predictions_folder, filename)
            fact_file_path = os.path.join(fact_folder, filename)

            score = {
                "entailment": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0
                },
                "contradiction": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0
                },
                "neutral": {
                    "fp": 0,
                }
            }

            # Read content from the text file
            with open(prediction_file_path, 'r') as file:
                prediction = file.read()
            
            with open(fact_file_path, 'r') as file:
                facts = [f.strip().split("\t") for f in file.read().split("\n") if f.strip()]

            # Iterate through each fact
            for fact in facts:
                entail_pred = check_implication(fact[0], prediction)
                entail_gold = fact[1]

                if entail_pred != entail_gold:
                    score[entail_pred]["fp"] += 1
                    score[entail_gold]["fn"] += 1
                else:
                    score[entail_pred]["tp"] += 1

            if len(facts) == 0:
                recalled_info_rate += 1.0
                this_file_info_rate += 1.0
                this_file_error_rate = 0.0
            else:
                recalled_info_rate +=  (score["entailment"]["tp"] + score["contradiction"]["tp"]) / len(facts)
                error_rate += (score["entailment"]["fp"] + score["contradiction"]["fp"]) / len(facts)

                this_file_info_rate = (score["entailment"]["tp"] + score["contradiction"]["tp"]) / len(facts)
                this_file_error_rate = (score["entailment"]["fp"] + score["contradiction"]["fp"]) / len(facts)

            print(f"{filename}\t{this_file_info_rate}\t{this_file_error_rate}")

    # If reading entire dataset, print averages
    if end_at == -1:
        print(recalled_info_rate)
        print(error_rate)
        print(recalled_info_rate / len(dataset_lines))
        print(error_rate / len(dataset_lines))

parser = argparse.ArgumentParser(description='Evaluate MBCs using atomic facts')
parser.add_argument('--start_at', type=int, default=0)
parser.add_argument('--end_at', type=int, default=-1)
parser.add_argument('--dataset_file', type=str, default="data/splits/dev.tsv")
parser.add_argument('--predictions_folder', type=str, default='results_search_llama')
parser.add_argument('--fact_folder', type=str, default="data/splits/dev_facts")
args = parser.parse_args()

# Example usage
dataset_file = args.dataset_file
predictions_folder = args.predictions_folder
fact_folder = args.fact_folder

process_files(predictions_folder, dataset_file, fact_folder, start_at=args.start_at, end_at=args.end_at)