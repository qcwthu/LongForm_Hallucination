from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()
import torch
import json
import os
import random
import numpy as np
import time
from typing import List
import matplotlib.pyplot as plt
from openai import OpenAI
import heapq

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hd(model, tokenizer, question_text_original, terminators):
    question_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_text_original}],
        add_generation_prompt=True,
        tokenize=False
    )
    question_text_ids = tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        all_hidden_states = model(question_text_ids, output_hidden_states=True).hidden_states
    # print(len(all_hidden_states))
    # print(all_hidden_states[0].shape)
    
    final_layer_final_token = all_hidden_states[-1][0][-1].clone().detach()

    all_layers_final_token = all_hidden_states[1][0][-1].clone().detach()
    for i in range(2, len(all_hidden_states)):
        all_layers_final_token += all_hidden_states[i][0][-1].clone().detach()
    all_layers_final_token = all_layers_final_token / (len(all_hidden_states) - 1)

    # print(all_hidden_states[-1][0][:].clone().detach().shape)
    final_layer_all_tokens = torch.mean(all_hidden_states[-1][0][:].clone().detach(), dim=0)

    final_layer_final_and_all_tokens = 0.5 * (final_layer_final_token + final_layer_all_tokens)

    # print(final_layer_final_token.shape, all_layers_final_token.shape, final_layer_all_tokens.shape, final_layer_final_and_all_tokens.shape)
    return final_layer_final_token.tolist(), all_layers_final_token.tolist(), final_layer_all_tokens.tolist(), final_layer_final_and_all_tokens.tolist()

def get_hd_noinstruction_onlyclaim(model, tokenizer, text, terminators):
    text_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        all_hidden_states = model(text_ids, output_hidden_states=True).hidden_states
    # print(len(all_hidden_states))
    # print(all_hidden_states[0].shape)
    
    final_layer_final_token = all_hidden_states[-1][0][-1].clone().detach()

    all_layers_final_token = all_hidden_states[1][0][-1].clone().detach()
    for i in range(2, len(all_hidden_states)):
        all_layers_final_token += all_hidden_states[i][0][-1].clone().detach()
    all_layers_final_token = all_layers_final_token / (len(all_hidden_states) - 1)

    # print(all_hidden_states[-1][0][:].clone().detach().shape)
    final_layer_all_tokens = torch.mean(all_hidden_states[-1][0][:].clone().detach(), dim=0)

    final_layer_final_and_all_tokens = 0.5 * (final_layer_final_token + final_layer_all_tokens)

    # print(final_layer_final_token.shape, all_layers_final_token.shape, final_layer_all_tokens.shape, final_layer_final_and_all_tokens.shape)
    return final_layer_final_token.tolist(), all_layers_final_token.tolist(), final_layer_all_tokens.tolist(), final_layer_final_and_all_tokens.tolist()

def get_hd_question_and_claim(model, tokenizer, question_and_claim, terminators):
    question_and_claim_ids = tokenizer(question_and_claim, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        all_hidden_states = model(question_and_claim_ids, output_hidden_states=True).hidden_states
    # print(len(all_hidden_states))
    # print(all_hidden_states[0].shape)
    
    final_layer_final_token = all_hidden_states[-1][0][-1].clone().detach()

    all_layers_final_token = all_hidden_states[1][0][-1].clone().detach()
    for i in range(2, len(all_hidden_states)):
        all_layers_final_token += all_hidden_states[i][0][-1].clone().detach()
    all_layers_final_token = all_layers_final_token / (len(all_hidden_states) - 1)

    # print(all_hidden_states[-1][0][:].clone().detach().shape)
    final_layer_all_tokens = torch.mean(all_hidden_states[-1][0][:].clone().detach(), dim=0)

    final_layer_final_and_all_tokens = 0.5 * (final_layer_final_token + final_layer_all_tokens)

    # print(final_layer_final_token.shape, all_layers_final_token.shape, final_layer_all_tokens.shape, final_layer_final_and_all_tokens.shape)
    return final_layer_final_token.tolist(), all_layers_final_token.tolist(), final_layer_all_tokens.tolist(), final_layer_final_and_all_tokens.tolist()

def getHiddenStates():
    set_seed(42)

    inputfile_1 = "../Knowledge-Boundary/longfact-ACE-decomposed-facts/longfact-allfacts-transform2question.json"
    inputfile_2 = "../Knowledge-Boundary/longfact-ACE-decomposed-facts/longfact-allfacts-except-200-transform2question.json"

    allfact_1 = {}
    with open(inputfile_1,'r') as f:
        allfact_1 = json.load(f)
    print(len(allfact_1))

    allfact_2 = {}
    with open(inputfile_2,'r') as f:
        allfact_2 = json.load(f)
    print(len(allfact_2))

    allfact_1.update(allfact_2)
    print(len(allfact_1))

    numofsupported = 0
    numofnotsupported = 0
    numofcontradicted = 0
    numofnoreference = 0
    numofsearcherror = 0
    for onekey in allfact_1.keys():
        numofsupported += len(allfact_1[onekey]["supported"])
        numofnotsupported += len(allfact_1[onekey]["not_supported"])
        numofcontradicted += len(allfact_1[onekey]["contradicted"])
        numofnoreference += len(allfact_1[onekey]["no_reference"])
        numofsearcherror += len(allfact_1[onekey]["search_error"])
    print("number of supported: ", numofsupported)
    print("number of not supported: ", numofnotsupported)
    print("number of contradicted: ", numofcontradicted)
    print("number of no reference: ", numofnoreference)
    print("number of search error: ", numofsearcherror)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    # model.double()             ####for getting same probability between generate and forward
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()

    allsize = 0
    allres = {}
    finalresultpath = "./SelfCheckGPT/selfcheckdata.json"
    for onekey in allfact_1.keys(): 
        oneprompt = onekey.split("[/INST]")[0].split("[INST]")[1].strip()
        # print(oneprompt)
        input_tokens_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": oneprompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        allsupported = allfact_1[onekey]["supported"]
        allcontradicted = allfact_1[onekey]["contradicted"]
   
        if len(allsupported) > 0:
            allsupported_question_text = allfact_1[onekey]["supported_question_text"]
            assert len(allsupported) == len(allsupported_question_text[0]) == len(allsupported_question_text[1])
            allsupported_idx = [i for i in range(len(allsupported))]
        
        if len(allcontradicted) > 0:
            allcontradicted_question_text = allfact_1[onekey]["contradicted_question_text"]
            assert len(allcontradicted) == len(allcontradicted_question_text[0]) == len(allcontradicted_question_text[1])
            allcontradicted_idx = [i for i in range(len(allcontradicted))]
        
        if len(allsupported) > 0 and len(allcontradicted) > 0:
            minsize = min(len(allsupported), len(allcontradicted))
            if minsize > 0:
                sampled_supported_idx = random.sample(allsupported_idx, minsize)
                sampled_contradicted_idx = random.sample(allcontradicted_idx, minsize)
                allsize += minsize
                for idx in range(minsize):
                    one_supported_idx = sampled_supported_idx[idx]
                    onesupported = allsupported[one_supported_idx]
                    allres[onesupported] = oneprompt

                    one_contradicted_idx = sampled_contradicted_idx[idx]
                    onecontradicted = allcontradicted[one_contradicted_idx]
                    allres[onecontradicted] = oneprompt
                    
    with open(finalresultpath, "w+") as f:
        json.dump(allres, f, indent=4, separators=(",", ": "))

getHiddenStates()
