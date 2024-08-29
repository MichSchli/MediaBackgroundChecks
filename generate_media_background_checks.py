from openai import OpenAI
import os
import tqdm
import time
from urllib.parse import urlparse
import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from spacy.lang.en import English
import json
from googleapiclient.discovery import build
from utils.html2lines import url2text

class MBCGenerator:

    model_name = "deepset/deberta-v3-large-squad2"
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    nlp = English()
    nlp.add_pipe("sentencizer")
    google_api_key = None
    search_engine_id = None
    local_model = False

    blacklist = [
        "jstor.org", # Blacklisted because their pdfs are not labelled as such, and clog up the download
        "facebook.com", # Blacklisted because only post titles can be scraped, but the scraper doesn't know this,
        "ftp.cs.princeton.edu", # Blacklisted because it hosts many large NLP corpora that keep showing up
        "nlp.cs.princeton.edu",
        "mediabiasfactcheck.com", # We don't want to retrieve the test data
        "ground.news" # Cites mediabiasfactcheck too often
    ]

    blacklist_files = [ # Blacklisted some additional NLP files that show up in search results and cause OOM errors
        "/glove.", 
        "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
        "https://web.mit.edu/adamrose/Public/googlelist"
    ]

    def __init__(self, local_model=False) -> None:
        self.model = "gpt-3.5-turbo"
        with open("config.json", "r") as f:
            config = json.load(f)
        
        openai_key = config["openai_key"]
        self.google_api_key = config["google_api_key"]
        self.search_engine_id = config["search_engine_id"]
        hf_access_token = config["hf_access_token"]

        self.client = OpenAI(
            api_key= openai_key
        )

        if local_model:
            self.local_model = True
            self.local_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            access_token = hf_access_token
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id, token=access_token)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_id,
                #torch_dtype=torch.bfloat16,
                #device_map="cpu", 
                token=access_token
                )
        else:
            self.local_model = False

    def get_domain_name(self, url):
        if '://' not in url:
            url = 'http://' + url

        domain = urlparse(url).netloc

        if domain.startswith("www."):
            return domain[4:]
        else:
            return domain

    def __google_search__(self, search_term, **kwargs):
        service = build("customsearch", "v1", developerKey=self.google_api_key)
        res = service.cse().list(q=search_term, cx=self.search_engine_id, **kwargs).execute()

        if "items" in res:
            return res['items']
        else:
            return []
        
    def send_message_local(self, message):
        input_ids = self.local_tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(self.local_model.device)
        
        terminators = [
            self.local_tokenizer.eos_token_id,
            self.local_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        
        outputs = self.local_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            )
        
        response = outputs[0][input_ids.shape[-1]:]
        response = self.local_tokenizer.decode(response, skip_special_tokens=True)

        return response


    def send_message(self, message):
        if self.local_model:
            return self.send_message_local(message)
        
        result = None
        retries = 0
        while result is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message
                )
                result = response.choices[0].message.content
            except:
                print(f"Failed to send message. Retrying (attempt {retries + 1})...", file=sys.stderr)
                delay = min(1 * (2 ** retries), 64)
                time.sleep(delay)
                retries += 1

        return result
    
    def get_answer_from_evidence(self, question, evidence, qa_model = qa_model, nlp = nlp):

        evidence = str(evidence)[:3000].strip()

        if len(evidence) == 0:
            return {
                "backing": "none",
                "answer": "unavailable",
            }
        
        QA_input = {
            'question': question,
            'context': evidence
        }

        response = qa_model(QA_input)
        if response["score"] < 0.2:
            return {
                "backing": "none",
                "answer": "unavailable",
            }
        answer = response["answer"]
        start_token = response["start"]
        end_token = response["end"]

        backing = []
        sections_to_retrieve = 0
        for section in nlp(evidence).sents:
            section_start = section.start_char
            section_end = section.end_char

            if start_token >= section_start:
                sections_to_retrieve = 1
            if sections_to_retrieve:
                backing.append(section.text)
            if end_token <= section_end:
                sections_to_retrieve = 0
        backing = " ".join(backing)

        if len(backing) == 0:
            print("WARNING: I failed at producing JSON", file=sys.stderr)
            return {
                "backing": "none",
                "answer": "unavailable",
            }


        result = json.dumps({
            "backing":backing,
            "answer": answer
        })

        try:            
            result = json.loads(result)
        except:
            print("WARNING: I failed at producing JSON", file=sys.stderr)
            return {
                "backing": "none",
                "answer": "unavailable",
            }

        # If we have no answer, or the answer is unavailable, we just stop here. TODO: We probably want to do some double checking.
        if "answer" not in result:
            result["answer"] = "unavailable"
        if result["answer"] == "unavailable":
            return result

        # Verify that the answer contains backing, and that the backing is actually in the original document:
        if "backing" not in result:
            result["answer"] = "unavailable"

        return result
    
    def process_search_results(self, results, search_string):
        for result in results:
            link = str(result["link"])

            domain = self.get_domain_name(link)

            if domain in self.blacklist:
                continue

            for b_file in self.blacklist_files:
                if b_file in link:
                    continue
        
            if link.endswith(".doc"):
                continue

            if link.endswith(".pdf"):
                continue

            # We cache all downloaded websites. For now, each searcher has one cache. If we use multiple searchers we may want to unify. We may also want to place a limit on how many documents we store.
            evidence = url2text(link)

            keep_lines = []
            for line in evidence.split("\n"):
                # Keep lines with 3 or more words
                if len(line.strip().split(" ")) < 3:
                    continue

                if line in ["Something went wrong. Wait a moment and try again.", 
                            "Please enable Javascript and refresh the page to continue"]:
                    continue
                
                keep_lines.append(line.strip())

            if len(keep_lines) > 0:
                out_doc = "\n".join(keep_lines)
                yield {"evidence": out_doc, "link": result["link"]}
                
    def get_google_search_results(self, search_string, sort_date=None, page=0):
        search_results = []
        sort = None if sort_date is None else ("date:r:19000101:"+sort_date)

        for _ in range(3):
            try:
                search_results += self.__google_search__(
                    search_string,
                    num=10,
                    start=0,
                    sort=sort,
                    dateRestrict=None,
                    gl="US"
                )
                break
            except:
                print("I encountered an error trying to search +\""+search_string+"\". Maybe the connection dropped. Trying again in 3 seconds...", file=sys.stderr)
                time.sleep(3)
        return self.process_search_results(search_results, search_string)
    
    def get_icl_examples(self, n=1):
        return [
            """**Background check for The Guardian**
About
Launched in 1821, The Guardian is a British daily newspaper published in London, UK. Its original name is The Manchester Guardian, and cotton merchant John Edward Taylor founded it. In 1993 the Guardian Media Group acquired the Observer.

The paper focuses on politics, policy, business, and international relations. Their coverage includes News and Opinion, Sports, Culture, Lifestyle, Podcasts, and more.

Funded by / Ownership
The Guardian and its sister publication, the Sunday newspaper The Observer, are owned by Guardian Media Group plc (GMG). Scott Trust Limited was created in 1936 to ensure the editorial independence of the publications and owns Guardian Media Group plc (GMG). The Guardian states that “The Scott Trust is the sole shareholder in Guardian Media Group, and its profits are reinvested in journalism and do not benefit a proprietor or shareholders.” Donations and advertising fund the Guardian.

Analysis / Bias
The Guardian has always been a left-wing publication throughout its history, as they have stated in various articles.
Story selection favors the left but is generally factual. They utilize emotionally loaded headlines such as “The cashless society is a con – and big finance is behind it” and “Trump back-pedals on Russian meddling remarks after an outcry.” 

A 2014 Pew Research Survey found that 72% of The Guardian’s audience is consistently or primarily liberal, 20% Mixed, and 9% consistently or mostly conservative. This indicates that a more liberal audience strongly prefers the Guardian. 

Failed Fact Checks
* Is everything you think you know about depression wrong? – False
* Firms bidding for government contracts asked if they back Brexit. – False
* The proportion of lung cancer cases only diagnosed after a visit to an A&E ranges from 15% in Guildford and Waverley in Surrey to 56% in Tower Hamlets and Manchester. – Inaccurate

            """,
            """**Background check for The New York Times**
About
The New York Times (sometimes abbreviated to NYT) is an American daily newspaper, founded and continuously published in New York City since September 18, 1851, by The New York Times Company.
The New York Times was initially founded by U.S. journalist and politician Henry Jarvis Raymond and former banker George Jones. 

Funded by / Ownership
The Ochs-Sulzberger family controls the New York Times through Class B shares. Since 1967, the company has been listed on the New York Stock Exchange under the symbol NYT. Class B shares are those that are held privately. The owner and publisher of the New York Times are The New York Times Company, and the Chairman is Arthur Gregg “A.G.” Sulzberger, succeeding his father Arthur Ochs Sulzberger Jr. He is the sixth member of the Ochs/Sulzberger family to serve as publisher since Adolph Ochs purchased the newspaper in 1896.
Mark Thompson became president and chief executive officer of The New York Times Company in 2012. Advertising and subscription fees generate revenue.

Analysis / Bias
The News York Times’ coverage includes News (World News, National News, Business News), Opinion Pieces, Editorials, Arts, Movies, Theater, Travel, NYC Guide, Food, Home & Garden, and  Fashion & Style.

The NYT looks at the issues from a progressive perspective and is regarded as “liberal.” According to a Pew Research Centers’ media polarization report, “the ideological Placement of Each Source’s Audience” places the audience for the New York Times as “consistently liberal.” Further, since 1960 The New York Times has only endorsed Democratic Presidential Candidates. 

Failed Fact Checks

* “We have a host of issues associated with high B.M.I.s. But correlation doesn’t prove causation, and there’s a significant body of research showing that weight stigma and weight cycling can explain most if not all of the associations we see between higher weights and poor health outcomes.” – MOSTLY FALSE
* A political map circulated by Sarah Palin’s PAC incited Rep. Gabby Giffords’ shooting – FALSE            
            """
            ][:n]
    
    def get_icl_string(self, n=1):
        return "The following are examples of background checks:\n\n" "\n\n".join(["\"" + example + "\"" for example in self.get_icl_examples(n)])

    def generate_initial_guess(self, source, use_icl=False):
        system_message = "You are InfoHuntGPT, a world-class AI assistant used by journalists to quickly build knowledge of new sources."
        prompt = "Build a background check for the news source \"" + source + "\". Write down everything you know about them, e.g. who funds them, how they make money, if they have any particular bias. Make an ITEMIZED LIST. Be brief, and if you don't know something, just leave it out.\n"
        prompt += "If you are aware that they have failed any fact-checks, mention which. Begin your response with \"**Background check**\"."

        if use_icl:
            prompt += "\n\n" + self.get_icl_string()

        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    
        result = self. send_message(messages)
        return result
    
    def incorporate_extra_information(self, source, previous_guess, new_information):
        system_message = "You are InfoHuntGPT, a world-class tool used by journalists to quickly build knowledge of new sources."
        initial_prompt = "Build a background check for the news source \"" + source + "\". Write down everything you know about them, e.g. who funds them, how they make money, if they have any particular bias. Make an ITEMIZED LIST. Be brief, and if you don't know something, just leave it out.\n"
        initial_prompt += "If you are aware that they have failed any fact-checks, mention which. Begin your response with \"**Background check**\"."
        bot_answer = previous_guess
        prompt = "Google search has revealed some new information:\n\n"
        prompt += new_information + "\n\n"
        prompt += "Update your background check for \"" + source + "\" using the new information. Do NOT delete any information, but make ADDITIONS where necessary, using the new information. Most likely, you will just need to add an extra item to the itemized list you previously created. Make minimal edits, and only incorporate what is relevant. Begin your response with \"**Background check**\""

        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": bot_answer},
            {"role": "user", "content": prompt},
        ]
    
        result = self.send_message(messages)
        return result

    def build_background_check(self, source, questions_file="data/queries.json", run_search=True, back_initial=True, use_icl=False):
        with open(questions_file, 'r') as file:
            data = json.load(file)

        initial_guess = self.generate_initial_guess(source, use_icl=use_icl)
        bgcheck = initial_guess

        if not run_search:
            # Can stop here for gpt-only version
            return bgcheck, None
        
        search_lines = []
        
        for item in data:
            localized_question = item['question'].strip().replace("X", "\"" + source + "\"")
            localized_search_query = item['statement'].strip().replace("X", "\"" + source + "\"")
            search_results = self.get_google_search_results(localized_search_query)
            extra_info = ""
            for s in search_results:
                s["question"] = localized_search_query
                search_lines.append(s)
                evidence = s["evidence"]
            
                answer = self.get_answer_from_evidence(localized_question, evidence)
                if answer["answer"] == "unavailable":
                    continue

                if len(extra_info) > 0:
                    extra_info += "\n"

                extra_info += localized_question.replace("\"", "") + " " + answer["backing"]

            if len(extra_info) > 0:
                bgcheck = self.incorporate_extra_information(source, bgcheck, extra_info)

        if back_initial:
            bgcheck = self.incorporate_extra_information(source, bgcheck, initial_guess)

        return bgcheck, search_lines

def process_tsv(input_file, output_folder, search=True, local_model=False):
    mbc_generator = MBCGenerator(local_model=local_model)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the file
    with open(input_file, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    # Iterate through each line
    pbar = tqdm.tqdm(lines, desc="Processing files")
    for line in pbar:
        test_source = line[0]  # Get the name from the first column
        pbar.set_description(f"Processing \"{test_source}\"")
        
        source_check, search_lines = mbc_generator.build_background_check(test_source.replace(".txt", ""), run_search=search)

        output_file = os.path.join(output_folder, test_source)
        with open(output_file, 'w') as f:
            f.write(source_check)

        if search_lines is not None:
            search_folder = os.path.join(output_folder, test_source.replace(".txt", "") + "_search")
            if not os.path.exists(search_folder):
                os.makedirs(search_folder)

            for i, search_line in enumerate(search_lines):
                output_file = os.path.join(search_folder, str(i))
                with open(output_file, 'w') as f:
                    f.write(search_line["question"] + "\n")
                    f.write(search_line["link"] + "\n")
                    f.write(search_line["evidence"])

if __name__ == "__main__":
    process_tsv("data/dataset/test.tsv", "results_search_gpt", search=True, local_model=False)

