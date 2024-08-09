from openai import OpenAI
import os
import tqdm
import time
import sys
from collections import Counter
import json

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

def send_message(message):
    model = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=message
    )
    text = response.choices[0].message.content

    return text

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

def process_files(questions_file, folder_path, dataset_file, output_folder, start_at=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(dataset_file, 'r') as f:
        dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()]

    # Load the JSON file
    with open(questions_file, 'r') as file:
        data = json.load(file)

    questions = [item['statement'].strip() for item in data]

    # Iterate through each text file in the folder
    print("Starting at " + str(start_at))
    for i, filename in enumerate(tqdm.tqdm(dataset_lines)):
        if i < start_at:
            pass
        else:
            file_path = os.path.join(folder_path, filename)

            # Read content from the text file
            with open(file_path, 'r') as file:
                text_content = file.read()

            parts = text_content.split("\nHistory\n")
            text_to_check = "History\n" + "\nHistory\n".join(parts[1:]) if len(parts) > 1 else ""
            text_to_check = text_to_check.replace("[Media Bias Fact Check]()", "").split("Last Updated")[0]
            text_to_check = text_to_check.replace("Mediabiasfactcheck.com", "")

            # Initialize a list to store results for each question and text combination
            atomic_facts = []
            outlet = filename[:-4]

            # Iterate through each question
            for question in questions:
                localized_question = question.replace("X", outlet)
                if "_" not in question:
                    result = localized_question
                else:
                    result = generate_atomic_fact(localized_question, text_to_check).replace("_", "")

                # Basic check: No newlines, no tabs, at least as many words as the original
                result = result.split("\n")[-1].replace("\t", " ")
                if len(result.split()) < len(question.split()):
                    continue

                entail_pos = check_implication(result, text_to_check)
                if entail_pos != "neutral":
                    atomic_facts.append("\t".join([result, entail_pos]))

            # Append the list of results for the current file to the main results list
            output_file = os.path.join(output_folder, filename)
            with open(output_file, 'w') as f:
                f.write("\n".join(atomic_facts))

start_at = 0
dataset_file = "dataset/test.tsv"
questions_file_path = 'data/queries.json'
data_folder_path = 'data_2'
output_folder = "dataset/test/atomic_facts"
process_files(questions_file_path, data_folder_path, dataset_file, output_folder, start_at=start_at)