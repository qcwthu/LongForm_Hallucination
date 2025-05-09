from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()
import torch
import torch.nn.functional as F
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import time
from typing import List
import matplotlib.pyplot as plt
import heapq
from sklearn.metrics import roc_auc_score
import gc
# from vllm import LLM, SamplingParams
from scipy.stats import pearsonr, spearmanr
import argparse
import csv
import pandas as pd
from openai import OpenAI

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data(data_path):
    
    all_hidden_states = json.load(open(data_path, encoding='utf-8'))

    final_layer_final_token = all_hidden_states["final_layer_final_token"]
    all_layers_final_token = all_hidden_states["all_layers_final_token"]
    final_layer_all_tokens = all_hidden_states["final_layer_all_tokens"]
    final_layer_final_and_all_tokens = all_hidden_states["final_layer_final_and_all_tokens"]
    all_support_text = all_hidden_states["all_support_text"]
    all_contradicted_text = all_hidden_states["all_contradicted_text"]
    
    assert len(final_layer_final_token) == len(all_layers_final_token) == len(final_layer_all_tokens) == len(final_layer_final_and_all_tokens)

    right_final_layer_final_token = []
    hallu_final_layer_final_token = []
    for onedata in final_layer_final_token:
        right_final_layer_final_token.append(onedata["right"])
        hallu_final_layer_final_token.append(onedata["hallu"])
    
    right_all_layers_final_token = []
    hallu_all_layers_final_token = []
    for onedata in all_layers_final_token:
        right_all_layers_final_token.append(onedata["right"])
        hallu_all_layers_final_token.append(onedata["hallu"])

    right_final_layer_all_tokens = []
    hallu_final_layer_all_tokens = []
    for onedata in final_layer_all_tokens:
        right_final_layer_all_tokens.append(onedata["right"])
        hallu_final_layer_all_tokens.append(onedata["hallu"])

    right_final_layer_final_and_all_tokens = []
    hallu_final_layer_final_and_all_tokens = []
    for onedata in final_layer_final_and_all_tokens:
        right_final_layer_final_and_all_tokens.append(onedata["right"])
        hallu_final_layer_final_and_all_tokens.append(onedata["hallu"])

    enddata = []
    for i in range(len(right_final_layer_final_token)):
        enddata.append({
            "fact": all_support_text[i]["fact"],
            "question": all_support_text[i]["question"],
            "hidden": right_final_layer_final_token[i],
            "label": 0
        })
        enddata.append({
            "fact": all_contradicted_text[i]["fact"],
            "question": all_contradicted_text[i]["question"],
            "hidden": hallu_final_layer_final_token[i],
            "label": 1
        })
        
    return enddata

def handle_decomposed_facts():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates_factscore/factscore_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
    all_data = get_data(data_path)
    random.shuffle(all_data)
    print(len(all_data))
    train_index = int(len(all_data) * train_split)
    valid_index = int(len(all_data) * (train_split + valid_split))
    print(train_index, valid_index)

    train_data = all_data[0:train_index]
    valid_data = all_data[train_index:valid_index]
    test_data = all_data[valid_index:]

    print(len(train_data), len(valid_data), len(test_data))
    # print(train_data[0]["fact"])
    # print(train_data[-1]["fact"])
    # print(valid_data[0]["fact"])
    # print(valid_data[-1]["fact"])
    # print(test_data[0]["fact"])
    # print(test_data[-1]["fact"])
    # exit -1

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    # exit -1
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    alltosavetrain = []
    for i in range(len(train_data)):
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        if train_data[i]["label"] == 0:
            # response = "True"
            response = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
        elif train_data[i]["label"] == 1:
            # response = "False"
            response = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
        else:
            exit -1
        onetosavetrain = {
            "instruction": instruction,
            "input": query,
            "output": response
        }
        alltosavetrain.append(onetosavetrain)
    
    alltosavevalid = []
    for i in range(len(valid_data)):
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        if valid_data[i]["label"] == 0:
            # response = "True"
            response = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
        elif valid_data[i]["label"] == 1:
            # response = "False"
            response = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
        else:
            exit -1
        onetosavevalid = {
            "instruction": instruction,
            "input": query,
            "output": response
        }
        alltosavevalid.append(onetosavevalid)
    print(len(alltosavetrain), len(alltosavevalid))
    print(alltosavetrain[0])
    print(alltosavevalid[0])
    
    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_CoT.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_CoT.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))


def handle_decomposed_facts_true_false_cot():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates_factscore/factscore_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
    all_data = get_data(data_path)
    random.shuffle(all_data)
    print(len(all_data))
    train_index = int(len(all_data) * train_split)
    valid_index = int(len(all_data) * (train_split + valid_split))
    print(train_index, valid_index)

    train_data = all_data[0:train_index]
    valid_data = all_data[train_index:valid_index]
    test_data = all_data[valid_index:]

    print(len(train_data), len(valid_data), len(test_data))
    # print(train_data[0]["fact"])
    # print(train_data[-1]["fact"])
    # print(valid_data[0]["fact"])
    # print(valid_data[-1]["fact"])
    # print(test_data[0]["fact"])
    # print(test_data[-1]["fact"])
    # exit -1

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    alltosavetrain = []
    for i in range(len(train_data)):
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        if train_data[i]["label"] == 0:
            # response = "True"
            response = "True. " + allfact2cot[onlyquery]
        elif train_data[i]["label"] == 1:
            # response = "False"
            response = "False. " + allfact2cot[onlyquery]
        else:
            exit -1
        onetosavetrain = {
            "instruction": instruction,
            "input": query,
            "output": response
        }
        alltosavetrain.append(onetosavetrain)
    
    alltosavevalid = []
    for i in range(len(valid_data)):
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        if valid_data[i]["label"] == 0:
            # response = "True"
            response = "True. " + allfact2cot[onlyquery]
        elif valid_data[i]["label"] == 1:
            # response = "False"
            response = "False. " + allfact2cot[onlyquery]
        else:
            exit -1
        onetosavevalid = {
            "instruction": instruction,
            "input": query,
            "output": response
        }
        alltosavevalid.append(onetosavevalid)
    print(len(alltosavetrain), len(alltosavevalid))
    print(alltosavetrain[0])
    print(alltosavevalid[0])
    
    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_true_false_CoT.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_true_false_CoT.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))

def callgpt4o(client, input2gpt):
    try:
        responsefromgpt = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content":input2gpt}],
            temperature=0,
            stop=None
        )
    except Exception as exc:
        print(exc)
        return "gpt4o calling error"
    time.sleep(1.0)
    extractedtext = responsefromgpt.choices[0].message.content.strip()
    return extractedtext

def handle_decomposed_facts_paraphrase():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates_factscore/factscore_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
    all_data = get_data(data_path)
    random.shuffle(all_data)
    print(len(all_data))
    train_index = int(len(all_data) * train_split)
    valid_index = int(len(all_data) * (train_split + valid_split))
    print(train_index, valid_index)

    train_data = all_data[0:train_index]
    valid_data = all_data[train_index:valid_index]
    test_data = all_data[valid_index:]

    print(len(train_data), len(valid_data), len(test_data))
    # print(train_data[0]["fact"])
    # print(train_data[-1]["fact"])
    # print(valid_data[0]["fact"])
    # print(valid_data[-1]["fact"])
    # print(test_data[0]["fact"])
    # print(test_data[-1]["fact"])
    # exit -1

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    api_key = ""
    client = OpenAI(api_key=api_key)
    # touseprompt = "Given a statement, generate a paraphrase for it. Make sure that your response has the same meaning with the original statement and follows the format 'Paraphrase: {{Paraphrase}}'"
    touseprompt = "Given a statement, generate a paraphrase for it. Make sure that your response follows the format 'Paraphrase: {{Paraphrase}}'"
    instruction_query = f"Your task is to determine the correctness of the given claim. When presented with a claim, reply with 'True' or 'False'. Make sure that your response is exactly 'True' or 'False' without any extra commentary whatsoever."
    instruction_cot_true_false = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
    instruction_true_false_cot = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
    alltosavetrain_query = []
    alltosavetrain_cot_true_false = []
    alltosavetrain_true_false_cot = []
    for i in range(len(train_data)):
        print(i)
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        #####get paraphrases for onlyquery and allfact2cot[onlyquery]
        query_para_input = touseprompt + "\n\nStatement: " + onlyquery
        query_para_output = callgpt4o(client, query_para_input)
        query_para_output = "Claim: " + query_para_output.replace("Paraphrase: ", "").replace("paraphrase: ", "")

        cot_para_input = touseprompt + "\n\nStatement: " + allfact2cot[onlyquery]
        cot_para_output = callgpt4o(client, cot_para_input)
        cot_para_output = cot_para_output.replace("Paraphrase: ", "").replace("paraphrase: ", "")
        
        if train_data[i]["label"] == 0:
            response_query = "True"
            response_query_para = "True"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_para = cot_para_output + f" Therefore, the answer is \\boxed{{True}}."
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_para = "True. " + cot_para_output
        elif train_data[i]["label"] == 1:
            response_query = "False"
            response_query_para = "False"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_para = cot_para_output + f" Therefore, the answer is \\boxed{{False}}."
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_para = "False. " + cot_para_output
        else:
            exit -1
        onetosavetrain_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavetrain_query_para = {
            "instruction": instruction_query,
            "input": query_para_output,
            "output": response_query_para
        }
        onetosavetrain_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavetrain_cot_true_false_para = {
            "instruction": instruction_cot_true_false,
            "input": query_para_output,
            "output": response_cot_true_false_para
        }
        onetosavetrain_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavetrain_true_false_cot_para = {
            "instruction": instruction_true_false_cot,
            "input": query_para_output,
            "output": response_true_false_cot_para
        }
        alltosavetrain_query.append(onetosavetrain_query)
        alltosavetrain_query.append(onetosavetrain_query_para)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false_para)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot_para)
    
    alltosavevalid_query = []
    alltosavevalid_cot_true_false = []
    alltosavevalid_true_false_cot = []
    for i in range(len(valid_data)):
        print(i)
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        #####get paraphrases for onlyquery and allfact2cot[onlyquery]
        query_para_input = touseprompt + "\n\nStatement: " + onlyquery
        query_para_output = callgpt4o(client, query_para_input)
        query_para_output = "Claim: " + query_para_output.replace("Paraphrase: ", "").replace("paraphrase: ", "")

        cot_para_input = touseprompt + "\n\nStatement: " + allfact2cot[onlyquery]
        cot_para_output = callgpt4o(client, cot_para_input)
        cot_para_output = cot_para_output.replace("Paraphrase: ", "").replace("paraphrase: ", "")
        
        if valid_data[i]["label"] == 0:
            response_query = "True"
            response_query_para = "True"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_para = cot_para_output + f" Therefore, the answer is \\boxed{{True}}."
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_para = "True. " + cot_para_output
        elif valid_data[i]["label"] == 1:
            response_query = "False"
            response_query_para = "False"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_para = cot_para_output + f" Therefore, the answer is \\boxed{{False}}."
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_para = "False. " + cot_para_output
        else:
            exit -1
        onetosavevalid_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavevalid_query_para = {
            "instruction": instruction_query,
            "input": query_para_output,
            "output": response_query_para
        }
        onetosavevalid_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavevalid_cot_true_false_para = {
            "instruction": instruction_cot_true_false,
            "input": query_para_output,
            "output": response_cot_true_false_para
        }
        onetosavevalid_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavevalid_true_false_cot_para = {
            "instruction": instruction_true_false_cot,
            "input": query_para_output,
            "output": response_true_false_cot_para
        }
        alltosavevalid_query.append(onetosavevalid_query)
        alltosavevalid_query.append(onetosavevalid_query_para)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false_para)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot_para)
        
    print(len(alltosavetrain_query), len(alltosavevalid_query))
    print(len(alltosavetrain_cot_true_false), len(alltosavevalid_cot_true_false))
    print(len(alltosavetrain_true_false_cot), len(alltosavevalid_true_false_cot))
    print(alltosavetrain_query[0])
    print(alltosavevalid_query[0])
    
    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_withpara.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_withpara.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_query, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_query, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_cot_true_false_withpara.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_cot_true_false_withpara.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_cot_true_false, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_cot_true_false, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_true_false_cot_withpara.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_true_false_cot_withpara.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_true_false_cot, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_true_false_cot, f, indent=4, separators=(",", ": "))


def handle_decomposed_facts_different_prompt():
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates_factscore/factscore_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
    all_data = get_data(data_path)
    random.shuffle(all_data)
    print(len(all_data))
    train_index = int(len(all_data) * train_split)
    valid_index = int(len(all_data) * (train_split + valid_split))
    print(train_index, valid_index)

    train_data = all_data[0:train_index]
    valid_data = all_data[train_index:valid_index]
    test_data = all_data[valid_index:]

    print(len(train_data), len(valid_data), len(test_data))
    # print(train_data[0]["fact"])
    # print(train_data[-1]["fact"])
    # print(valid_data[0]["fact"])
    # print(valid_data[-1]["fact"])
    # print(test_data[0]["fact"])
    # print(test_data[-1]["fact"])
    # exit -1

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    api_key = ""
    client = OpenAI(api_key=api_key)

    touseprompt_support = "Given a correct claim and why it is correct, first identity the key information in the claim, then transform it into a question and a correct answer (keep the answer as concise as possible) about the key information, finally give the explanation (keep it different from the given reason). Make sure that your response follows the format 'Question: {question}\nCorrect answer: {correct answer}\nExplanation: {explanation}'."
    touseprompt_contradicted = "Given a wrong claim and why it is wrong, first identity the key information in the claim, then transform it into a question and a wrong answer (keep the answer as concise as possible) about the key information, finally give the explanation (keep it different from the given reason). Make sure that your response follows the format 'Question: {question}\nWrong answer: {wrong answer}\nExplanation: {explanation}'."

    for i in range(len(train_data)):
        print(i)
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        
        if train_data[i]["label"] == 0:
            fullprompt = touseprompt_support + "\n\nCorrect claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            train_data[i]["gpt_response"] = gpt_response
        elif train_data[i]["label"] == 1:
            fullprompt = touseprompt_contradicted + "\n\nWrong claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            train_data[i]["gpt_response"] = gpt_response
        else:
            exit -1
        train_data[i]["hidden"] = []
    ####save new train_data
    trainpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_train_different_prompt.json"
    with open(trainpath, "w+") as f:
        json.dump(train_data, f, indent=4, separators=(",", ": "))
    
    for i in range(len(valid_data)):
        print(i)
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        
        if valid_data[i]["label"] == 0:
            fullprompt = touseprompt_support + "\n\nCorrect claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            valid_data[i]["gpt_response"] = gpt_response
        elif valid_data[i]["label"] == 1:
            fullprompt = touseprompt_contradicted + "\n\nWrong claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            valid_data[i]["gpt_response"] = gpt_response
        else:
            exit -1
        valid_data[i]["hidden"] = []
    validpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_valid_different_prompt.json"
    with open(validpath, "w+") as f:
        json.dump(valid_data, f, indent=4, separators=(",", ": "))


def handle_decomposed_facts_prompt2llamadata():
    set_seed(42)
    
    trainpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_train_different_prompt.json"
    with open(trainpath,'r') as f:
        train_data = json.load(f)

    validpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_valid_different_prompt.json"
    with open(validpath,'r') as f:
        valid_data = json.load(f)

    print(len(train_data))
    print(len(valid_data))

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    instruction_query = f"Your task is to determine the correctness of the given claim. When presented with a claim, reply with 'True' or 'False'. Make sure that your response is exactly 'True' or 'False' without any extra commentary whatsoever."
    instruction_cot_true_false = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
    instruction_true_false_cot = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
    
    instruction_query_new_prompt = f"Answer the following question with 'Yes' or 'No'."
    instruction_cot_true_false_new_prompt = "Answer the following question with 'Yes' or 'No' and provide the explanation."
    instruction_true_false_cot_new_prompt = "Answer the following question with 'Yes' or 'No' and provide the explanation."


    alltosavetrain_query = []
    alltosavetrain_cot_true_false = []
    alltosavetrain_true_false_cot = []
    for i in range(len(train_data)):
        print(i)
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        #####get responses for new prompts
        gpt_response = train_data[i]["gpt_response"]
        # print(gpt_response)
        assert "Question: " in gpt_response
        if train_data[i]["label"] == 0:
            assert "Correct answer: " in gpt_response
        else:
            assert "Wrong answer: " in gpt_response
        assert "Explanation: " in gpt_response
        if train_data[i]["label"] == 0:
            new_question = gpt_response.split("Question: ")[-1].split("Correct answer: ")[0].strip()
            new_answer = gpt_response.split("Correct answer: ")[-1].split("Explanation: ")[0].strip()
            new_explanation = gpt_response.split("Explanation: ")[-1].strip()
        else:
            new_question = gpt_response.split("Question: ")[-1].split("Wrong answer: ")[0].strip()
            new_answer = gpt_response.split("Wrong answer: ")[-1].split("Explanation: ")[0].strip()
            new_explanation = gpt_response.split("Explanation: ")[-1].strip()

        query_new_prompt_output = "Question: Is '" + new_answer + "' the correct answer to the question '" + new_question + "'?"
        
        if train_data[i]["label"] == 0:
            response_query = "True"
            response_query_new_prompt = "Yes"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_new_prompt = new_explanation + f" So the answer to the question is 'Yes'."
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Yes. " + new_explanation
        elif train_data[i]["label"] == 1:
            response_query = "False"
            response_query_new_prompt = "No"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_new_prompt = new_explanation + f" So the answer to the question is 'No'."
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "No. " + new_explanation
        else:
            exit -1
        onetosavetrain_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavetrain_query_new_prompt = {
            "instruction": instruction_query_new_prompt,
            "input": query_new_prompt_output,
            "output": response_query_new_prompt
        }
        onetosavetrain_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavetrain_cot_true_false_new_prompt = {
            "instruction": instruction_cot_true_false_new_prompt,
            "input": query_new_prompt_output,
            "output": response_cot_true_false_new_prompt
        }
        onetosavetrain_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavetrain_true_false_cot_new_prompt = {
            "instruction": instruction_true_false_cot_new_prompt,
            "input": query_new_prompt_output,
            "output": response_true_false_cot_new_prompt
        }
       
        alltosavetrain_query.append(onetosavetrain_query)
        alltosavetrain_query.append(onetosavetrain_query_new_prompt)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false_new_prompt)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot_new_prompt)

    alltosavevalid_query = []
    alltosavevalid_cot_true_false = []
    alltosavevalid_true_false_cot = []
    for i in range(len(valid_data)):
        print(i)
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        #####get responses for new prompts
        gpt_response = valid_data[i]["gpt_response"]
        # print(gpt_response)
        assert "Question: " in gpt_response
        if valid_data[i]["label"] == 0:
            assert "Correct answer: " in gpt_response
        else:
            assert "Wrong answer: " in gpt_response
        assert "Explanation: " in gpt_response
        if valid_data[i]["label"] == 0:
            new_question = gpt_response.split("Question: ")[-1].split("Correct answer: ")[0].strip()
            new_answer = gpt_response.split("Correct answer: ")[-1].split("Explanation: ")[0].strip()
            new_explanation = gpt_response.split("Explanation: ")[-1].strip()
        else:
            new_question = gpt_response.split("Question: ")[-1].split("Wrong answer: ")[0].strip()
            new_answer = gpt_response.split("Wrong answer: ")[-1].split("Explanation: ")[0].strip()
            new_explanation = gpt_response.split("Explanation: ")[-1].strip()
        
        query_new_prompt_output = "Question: Is '" + new_answer + "' the correct answer to the question '" + new_question + "'?"

        if valid_data[i]["label"] == 0:
            response_query = "True"
            response_query_new_prompt = "Yes"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_new_prompt = new_explanation + f" So the answer to the question is 'Yes'."
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Yes. " + new_explanation
        elif valid_data[i]["label"] == 1:
            response_query = "False"
            response_query_new_prompt = "No"
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_new_prompt = new_explanation + f" So the answer to the question is 'No'."
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "No. " + new_explanation
        else:
            exit -1
        onetosavevalid_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavevalid_query_new_prompt = {
            "instruction": instruction_query_new_prompt,
            "input": query_new_prompt_output,
            "output": response_query_new_prompt
        }
        onetosavevalid_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavevalid_cot_true_false_new_prompt = {
            "instruction": instruction_cot_true_false_new_prompt,
            "input": query_new_prompt_output,
            "output": response_cot_true_false_new_prompt
        }
        onetosavevalid_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavevalid_true_false_cot_new_prompt = {
            "instruction": instruction_true_false_cot_new_prompt,
            "input": query_new_prompt_output,
            "output": response_true_false_cot_new_prompt
        }

        alltosavevalid_query.append(onetosavevalid_query)
        alltosavevalid_query.append(onetosavevalid_query_new_prompt)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false_new_prompt)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot_new_prompt)
        
    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_diff_prompt.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_diff_prompt.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_query, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_query, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_cot_true_false_diff_prompt.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_cot_true_false_diff_prompt.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_cot_true_false, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_cot_true_false, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_true_false_cot_diff_prompt.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_true_false_cot_diff_prompt.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_true_false_cot, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_true_false_cot, f, indent=4, separators=(",", ": "))

def handle_decomposed_facts_different_prompt_qa_format():
    set_seed(42)
    
    trainpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_train_different_prompt.json"
    with open(trainpath,'r') as f:
        train_data = json.load(f)

    validpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_valid_different_prompt.json"
    with open(validpath,'r') as f:
        valid_data = json.load(f)

    print(len(train_data))
    print(len(valid_data))

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))


    api_key = ""
    client = OpenAI(api_key=api_key)

    touseprompt_contradicted = "Given a wrong claim and why it is wrong, first identity the key information in the claim, then transform it into a question and a correct answer (keep the answer as concise as possible) about the key information, finally give the explanation (keep it different from the given reason). Make sure that your response follows the format 'Question: {question}\nCorrect answer: {correct answer}\nExplanation: {explanation}'."

    for i in range(len(train_data)):
        print(i)
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        
        if train_data[i]["label"] == 0:
            continue
        elif train_data[i]["label"] == 1:
            fullprompt = touseprompt_contradicted + "\n\nWrong claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            train_data[i]["gpt_response"] = gpt_response
        else:
            exit -1
        train_data[i]["hidden"] = []
    ####save new train_data
    trainpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_train_different_prompt_qa_format.json"
    with open(trainpath, "w+") as f:
        json.dump(train_data, f, indent=4, separators=(",", ": "))

    for i in range(len(valid_data)):
        print(i)
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        
        if valid_data[i]["label"] == 0:
            continue
        elif valid_data[i]["label"] == 1:
            fullprompt = touseprompt_contradicted + "\n\nWrong claim: " + onlyquery + "\nReason: " + cot
            gpt_response = callgpt4o(client, fullprompt)
            valid_data[i]["gpt_response"] = gpt_response
        else:
            exit -1
        valid_data[i]["hidden"] = []
    validpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_valid_different_prompt_qa_format.json"
    with open(validpath, "w+") as f:
        json.dump(valid_data, f, indent=4, separators=(",", ": "))

def handle_decomposed_facts_different_prompt_qa_format_llamadata():
    set_seed(42)
    
    trainpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_train_different_prompt_qa_format.json"
    with open(trainpath,'r') as f:
        train_data = json.load(f)

    validpath = "/data/home/Knowledge-Finetune/DifferentPrompt/factscore_valid_different_prompt_qa_format.json"
    with open(validpath,'r') as f:
        valid_data = json.load(f)

    print(len(train_data))
    print(len(valid_data))

    resultsall = pd.read_csv('/data/home/Knowledge-Boundary/factscore/ACE-decomposed-facts/factscore_ace.csv')
    print(len(resultsall))
    alljudgements = resultsall["judgments"].apply(lambda x: json.loads(x))
    print(len(alljudgements))
    allfact2cot = {}
    for i in range(len(alljudgements)):
        currentjudgement = alljudgements[i]
        search_factuality_check = currentjudgement["all_criteria"]["accuracy"]["search_factuality_check"]
        for onefact in search_factuality_check.keys():
            # print("**********************************************************")
            # print(onefact)
            cot = search_factuality_check[onefact]["feedback"]
            tousecot = cot.split('"reasoning": ')[-1].split('"answer": ')[0].strip().strip('",').strip('"')
            # print(tousecot)
            # print("----------------------------------------------------------")
            allfact2cot[onefact] = tousecot
    print(len(allfact2cot))

    instruction_query = f"Your task is to determine the correctness of the given claim. When presented with a claim, reply with 'True' or 'False'. Make sure that your response is exactly 'True' or 'False' without any extra commentary whatsoever."
    instruction_cot_true_false = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
    instruction_true_false_cot = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
    
    instruction_query_new_prompt = f"Answer the following question."
    instruction_cot_true_false_new_prompt = "Answer the following question and provide the explanation."
    instruction_true_false_cot_new_prompt = "Answer the following question and provide the explanation."


    alltosavetrain_query = []
    alltosavetrain_cot_true_false = []
    alltosavetrain_true_false_cot = []
    for i in range(len(train_data)):
        print(i)
        onedata = train_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        #####get responses for new prompts
        gpt_response = train_data[i]["gpt_response"]
        # print(gpt_response)
        assert "Question: " in gpt_response
        assert "Correct answer: " in gpt_response
        assert "Explanation: " in gpt_response
        
        new_question = gpt_response.split("Question: ")[-1].split("Correct answer: ")[0].strip()
        new_answer = gpt_response.split("Correct answer: ")[-1].split("Explanation: ")[0].strip()
        new_explanation = gpt_response.split("Explanation: ")[-1].strip()

        query_new_prompt_output = "Question: " + new_question
        
        if train_data[i]["label"] == 0:
            response_query = "True"
            response_query_new_prompt = "Answer: " + new_answer
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_new_prompt = "Explanation: " + new_explanation + f"\nAnswer: " + new_answer
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Answer: " + new_answer + "\nExplanation: " + new_explanation
        elif train_data[i]["label"] == 1:
            response_query = "False"
            response_query_new_prompt = "Answer: " + new_answer
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_new_prompt = "Explanation: " + new_explanation + f"\nAnswer: " + new_answer
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Answer: " + new_answer + "\nExplanation: " + new_explanation
        else:
            exit -1
        onetosavetrain_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavetrain_query_new_prompt = {
            "instruction": instruction_query_new_prompt,
            "input": query_new_prompt_output,
            "output": response_query_new_prompt
        }
        onetosavetrain_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavetrain_cot_true_false_new_prompt = {
            "instruction": instruction_cot_true_false_new_prompt,
            "input": query_new_prompt_output,
            "output": response_cot_true_false_new_prompt
        }
        onetosavetrain_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavetrain_true_false_cot_new_prompt = {
            "instruction": instruction_true_false_cot_new_prompt,
            "input": query_new_prompt_output,
            "output": response_true_false_cot_new_prompt
        }
        # print(onetosavetrain_query)
        # print("---------------------------------")
        # print(onetosavetrain_query_new_prompt)
        # print("---------------------------------")
        # print(onetosavetrain_cot_true_false)
        # print("---------------------------------")
        # print(onetosavetrain_cot_true_false_new_prompt)
        # print("---------------------------------")
        # print(onetosavetrain_true_false_cot)
        # print("---------------------------------")
        # print(onetosavetrain_true_false_cot_new_prompt)
        # print("---------------------------------")
        # exit -1
       
        alltosavetrain_query.append(onetosavetrain_query)
        alltosavetrain_query.append(onetosavetrain_query_new_prompt)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false)
        alltosavetrain_cot_true_false.append(onetosavetrain_cot_true_false_new_prompt)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot)
        alltosavetrain_true_false_cot.append(onetosavetrain_true_false_cot_new_prompt)

    alltosavevalid_query = []
    alltosavevalid_cot_true_false = []
    alltosavevalid_true_false_cot = []
    for i in range(len(valid_data)):
        print(i)
        onedata = valid_data[i]["question"].strip().split("\n")
        assert len(onedata) == 4
        # instruction = onedata[0]
        query = onedata[2]
        onlyquery = query.replace("Claim: ", "")
        assert onlyquery in allfact2cot.keys()
        cot = allfact2cot[onlyquery]
        #####get responses for new prompts
        gpt_response = valid_data[i]["gpt_response"]
        # print(gpt_response)
        assert "Question: " in gpt_response
        assert "Correct answer: " in gpt_response
        assert "Explanation: " in gpt_response
        
        new_question = gpt_response.split("Question: ")[-1].split("Correct answer: ")[0].strip()
        new_answer = gpt_response.split("Correct answer: ")[-1].split("Explanation: ")[0].strip()
        new_explanation = gpt_response.split("Explanation: ")[-1].strip()
        
        query_new_prompt_output = "Question: " + new_question

        if valid_data[i]["label"] == 0:
            response_query = "True"
            response_query_new_prompt = "Answer: " + new_answer
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{True}}."
            response_cot_true_false_new_prompt = "Explanation: " + new_explanation + f"\nAnswer: " + new_answer
            response_true_false_cot = "True. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Answer: " + new_answer + "\nExplanation: " + new_explanation
        elif valid_data[i]["label"] == 1:
            response_query = "False"
            response_query_new_prompt = "Answer: " + new_answer
            response_cot_true_false = allfact2cot[onlyquery] + f" Therefore, the answer is \\boxed{{False}}."
            response_cot_true_false_new_prompt = "Explanation: " + new_explanation + f"\nAnswer: " + new_answer
            response_true_false_cot = "False. " + allfact2cot[onlyquery]
            response_true_false_cot_new_prompt = "Answer: " + new_answer + "\nExplanation: " + new_explanation
        else:
            exit -1
        onetosavevalid_query = {
            "instruction": instruction_query,
            "input": query,
            "output": response_query
        }
        onetosavevalid_query_new_prompt = {
            "instruction": instruction_query_new_prompt,
            "input": query_new_prompt_output,
            "output": response_query_new_prompt
        }
        onetosavevalid_cot_true_false = {
            "instruction": instruction_cot_true_false,
            "input": query,
            "output": response_cot_true_false
        }
        onetosavevalid_cot_true_false_new_prompt = {
            "instruction": instruction_cot_true_false_new_prompt,
            "input": query_new_prompt_output,
            "output": response_cot_true_false_new_prompt
        }
        onetosavevalid_true_false_cot = {
            "instruction": instruction_true_false_cot,
            "input": query,
            "output": response_true_false_cot
        }
        onetosavevalid_true_false_cot_new_prompt = {
            "instruction": instruction_true_false_cot_new_prompt,
            "input": query_new_prompt_output,
            "output": response_true_false_cot_new_prompt
        }

        alltosavevalid_query.append(onetosavevalid_query)
        alltosavevalid_query.append(onetosavevalid_query_new_prompt)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false)
        alltosavevalid_cot_true_false.append(onetosavevalid_cot_true_false_new_prompt)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot)
        alltosavevalid_true_false_cot.append(onetosavevalid_true_false_cot_new_prompt)

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_diff_prompt_qa_format.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_diff_prompt_qa_format.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_query, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_query, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_cot_true_false_diff_prompt_qa_format.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_cot_true_false_diff_prompt_qa_format.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_cot_true_false, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_cot_true_false, f, indent=4, separators=(",", ": "))

    trainpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_train_true_false_cot_diff_prompt_qa_format.json"
    validpath = "/data/home/LLaMA-Factory/data/factscore_automized_claim_valid_true_false_cot_diff_prompt_qa_format.json"
    with open(trainpath, "w+") as f:
        json.dump(alltosavetrain_true_false_cot, f, indent=4, separators=(",", ": "))
    with open(validpath, "w+") as f:
        json.dump(alltosavevalid_true_false_cot, f, indent=4, separators=(",", ": "))

if __name__ == '__main__':
    # handle_decomposed_facts()
    # handle_decomposed_facts_true_false_cot()
    
    # handle_decomposed_facts_paraphrase()

    # handle_decomposed_facts_different_prompt()
    # handle_decomposed_facts_prompt2llamadata()

    # handle_decomposed_facts_different_prompt_qa_format()
    handle_decomposed_facts_different_prompt_qa_format_llamadata()
