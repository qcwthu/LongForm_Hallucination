from transformers import AutoTokenizer, AutoModelForCausalLM
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

def handleonefact(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, curfact):
    # ####prompt with facts
    # onepromptwithfact_tokens = input_tokens_chat + curfact
    # onepromptwithfact_ids = tokenizer(onepromptwithfact_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    # with torch.no_grad():
    #     alloutput = model(onepromptwithfact_ids, output_attentions=True)
    # logitoutput = alloutput.logits
    # tokenprobs = logitoutput[0][input_ids_chat.shape[-1]-1:-1]
    # tokenprobs_sm = torch.nn.functional.softmax(tokenprobs, dim=-1)

    # # max_probs, max_ids = torch.topk(tokenprobs_sm, 1, dim=-1)         #####note that curfact (might) be the original text generated by the model
    # # print(max_probs, max_ids)
    # # print(curfact)
    # # print(tokenizer.decode(max_ids.squeeze()))

    # inputsize = input_ids_chat.shape[-1]
    # allprobs = []
    # for tokenidx in range(tokenprobs_sm.shape[0]):
    #     oneprob = tokenprobs_sm[tokenidx][onepromptwithfact_ids[0][inputsize + tokenidx]].detach().cpu().item()
    #     allprobs.append(oneprob)
    # print(curfact)
    # print(allprobs)
    
    # ####only facts
    # onefact_ids = tokenizer(curfact, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    # with torch.no_grad():
    #     alloutputonlyfact = model(onefact_ids, output_attentions=True)
    # logitoutputonlyfact = alloutputonlyfact.logits
    # tokenprobsonlyfact = logitoutputonlyfact[0][:-1]
    # tokenprobsonlyfact_sm = torch.nn.functional.softmax(tokenprobsonlyfact, dim=-1)
    # # print(tokenprobsonlyfact_sm.shape)
    # allprobsonlyfact = []
    # for tokenidx in range(tokenprobsonlyfact_sm.shape[0]):
    #     oneprobonlyfact = tokenprobsonlyfact_sm[tokenidx][onefact_ids[0][tokenidx + 1]].detach().cpu().item()
    #     allprobsonlyfact.append(oneprobonlyfact)
    # print(curfact)
    # print(allprobsonlyfact)
    

    ###original full response
    onepromptwithresponse_tokens = input_tokens_chat + oneresponse
    onepromptwithresponse_ids = tokenizer(onepromptwithresponse_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        alloutputwithresponse = model(onepromptwithresponse_ids, output_attentions=True)
    logitoutputwithresponse = alloutputwithresponse.logits
    tokenprobswithresponse = logitoutputwithresponse[0][input_ids_chat.shape[-1]-1:-1]
    tokenprobswithresponse_sm = torch.nn.functional.softmax(tokenprobswithresponse, dim=-1)

    inputsize = input_ids_chat.shape[-1]
    allprobswithresponse = []
    for tokenidx in range(tokenprobswithresponse_sm.shape[0]):
        oneprobwithresponse = tokenprobswithresponse_sm[tokenidx][onepromptwithresponse_ids[0][inputsize + tokenidx]].detach().cpu().item()
        allprobswithresponse.append(oneprobwithresponse)
    print(allprobswithresponse)
    print(len(allprobswithresponse))

def average_of_min_k(lst, k):
    return sum(heapq.nsmallest(k, lst)) / k

def geometric_mean(lst):
    return np.exp(np.mean(np.log(lst)))

def getprobability(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, curfact, curfact_original_text):
    ####full response
    onepromptwithresponse_tokens = input_tokens_chat + oneresponse
    onepromptwithresponse_ids = tokenizer(onepromptwithresponse_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        alloutputwithresponse = model(onepromptwithresponse_ids, output_attentions=True)

    ####try to find the position of curfact_original_text in oneresponse
    textindex = oneresponse.find(curfact_original_text)
    assert textindex != -1
    firstpart = oneresponse[0:textindex]    #####maybe add .strip(' ')?
    secondpart = oneresponse[textindex:textindex+len(curfact_original_text)]
    assert secondpart == curfact_original_text
    
    promptwithfirstpart_tokens = input_tokens_chat + firstpart.strip(' ')
    promptwithbothpart_tokens = input_tokens_chat + firstpart + secondpart
    promptwithfirstpart_ids = tokenizer(promptwithfirstpart_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    promptwithbothpart_ids = tokenizer(promptwithbothpart_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    # print(tokenizer.decode(promptwithbothpart_ids[0][promptwithfirstpart_ids.shape[-1]:promptwithbothpart_ids.shape[-1]]))
    ####from promptwithfirstpart_ids.shape[-1] to promptwithbothpart_ids.shape[-1]
    logitoutputwithresponse = alloutputwithresponse.logits
    tokenprobswithresponse = logitoutputwithresponse[0][promptwithfirstpart_ids.shape[-1] - 1: promptwithbothpart_ids.shape[-1] - 1]
    tokenprobswithresponse_sm = torch.nn.functional.softmax(tokenprobswithresponse, dim=-1)
    
    inputsize = promptwithfirstpart_ids.shape[-1]
    allprobswithresponse = []
    for tokenidx in range(tokenprobswithresponse_sm.shape[0]):
        oneprobwithresponse = tokenprobswithresponse_sm[tokenidx][onepromptwithresponse_ids[0][inputsize + tokenidx]].detach().cpu().item()
        allprobswithresponse.append(oneprobwithresponse)
    # print(allprobswithresponse)
    print(len(allprobswithresponse))
    # return sum(allprobswithresponse) / len(allprobswithresponse)    ####arithmetic mean
    # return geometric_mean(allprobswithresponse)    ####geometric mean
    # return min(allprobswithresponse)
    # return average_of_min_k(allprobswithresponse, 3)
    # return average_of_min_k(allprobswithresponse, 5)
    # tosamplenumber = max(1, int(len(allprobswithresponse) * 0.05))
    # tosamplenumber = max(1, int(len(allprobswithresponse) * 0.10))
    # tosamplenumber = max(1, int(len(allprobswithresponse) * 0.15))
    # return average_of_min_k(allprobswithresponse, tosamplenumber)

    arithmetic_mean_prob = sum(allprobswithresponse) / len(allprobswithresponse)    ####arithmetic mean
    geometric_mean_prob = geometric_mean(allprobswithresponse)    ####geometric mean
    min_prob = min(allprobswithresponse)
    min_3_prob =  average_of_min_k(allprobswithresponse, 3)
    min_5_prob =  average_of_min_k(allprobswithresponse, 5)
    tosamplenumber_5percent = max(1, int(len(allprobswithresponse) * 0.05))
    tosamplenumber_10percent = max(1, int(len(allprobswithresponse) * 0.10))
    tosamplenumber_15percent = max(1, int(len(allprobswithresponse) * 0.15))
    min_5percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_5percent)
    min_10percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_10percent)
    min_15percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_15percent)
    return [arithmetic_mean_prob, geometric_mean_prob, min_prob, min_3_prob, min_5_prob, min_5percent_prob, min_10percent_prob, min_15percent_prob]


def getprobabilityentity(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, curfact, curfact_original_text, entities):
    # print(curfact_original_text)
    # print(entities)
    ####full response
    onepromptwithresponse_tokens = input_tokens_chat + oneresponse
    onepromptwithresponse_ids = tokenizer(onepromptwithresponse_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        alloutputwithresponse = model(onepromptwithresponse_ids, output_attentions=True)

    ####try to find the position of curfact_original_text in oneresponse
    textindex = oneresponse.find(curfact_original_text)
    assert textindex != -1
    firstpart = oneresponse[0:textindex]    #####maybe add .strip(' ')?
    secondpart = oneresponse[textindex:textindex+len(curfact_original_text)]
    assert secondpart == curfact_original_text
    # print(secondpart)
    
    allentities = entities.split(',')
    # print(allentities)
    allprobswithresponse = []
    for i in range(0, len(allentities)):
        oneentity = allentities[i]
        entityindex = curfact_original_text.find(oneentity)
        if entityindex ==  -1:
            continue
        assert entityindex != -1
        if entityindex == 0:
            promptwithfirstpart_tokens = input_tokens_chat + firstpart.strip(' ')
        else:
            promptwithfirstpart_tokens = input_tokens_chat + firstpart + curfact_original_text[0:entityindex].strip(' ')
        promptwithbothpart_tokens = input_tokens_chat + firstpart + curfact_original_text[0:entityindex] + curfact_original_text[entityindex:entityindex+len(oneentity)]
        promptwithfirstpart_ids = tokenizer(promptwithfirstpart_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        promptwithbothpart_ids = tokenizer(promptwithbothpart_tokens, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        promptentity_ids = tokenizer(" " + curfact_original_text[entityindex:entityindex+len(oneentity)], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

        logitoutputwithresponse = alloutputwithresponse.logits
        tokenprobswithresponse = logitoutputwithresponse[0][promptwithfirstpart_ids.shape[-1] - 1: promptwithbothpart_ids.shape[-1] - 1]
        tokenprobswithresponse_sm = torch.nn.functional.softmax(tokenprobswithresponse, dim=-1)

        inputsize = promptwithfirstpart_ids.shape[-1]
        
        for tokenidx in range(tokenprobswithresponse_sm.shape[0]):
            oneprobwithresponse = tokenprobswithresponse_sm[tokenidx][onepromptwithresponse_ids[0][inputsize + tokenidx]].detach().cpu().item()
            allprobswithresponse.append(oneprobwithresponse)
        # print(allprobswithresponse)

    print(len(allprobswithresponse))
    arithmetic_mean_prob = sum(allprobswithresponse) / len(allprobswithresponse)    ####arithmetic mean
    geometric_mean_prob = geometric_mean(allprobswithresponse)    ####geometric mean
    min_prob = min(allprobswithresponse)
    min_3_prob =  average_of_min_k(allprobswithresponse, 3)
    min_5_prob =  average_of_min_k(allprobswithresponse, 5)
    tosamplenumber_5percent = max(1, int(len(allprobswithresponse) * 0.05))
    tosamplenumber_10percent = max(1, int(len(allprobswithresponse) * 0.10))
    tosamplenumber_15percent = max(1, int(len(allprobswithresponse) * 0.15))
    min_5percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_5percent)
    min_10percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_10percent)
    min_15percent_prob = average_of_min_k(allprobswithresponse, tosamplenumber_15percent)
    return [arithmetic_mean_prob, geometric_mean_prob, min_prob, min_3_prob, min_5_prob, min_5percent_prob, min_10percent_prob, min_15percent_prob]

def callgpt4o(client, oneresponse, curfact):
    
    #####figure out the original sentence / atomic claim of the given fact
    # print("original response: ", oneresponse)
    # print("decomposed claim:", curfact)
    instruction = "Your task is to extract the original text corresponding to the given claim from the original response. When presented with an original response and a claim, reply with the original text. Make sure that your response is exactly the same as the original text and enclosed in \\boxed{}."
    # instruction = "Your task is to extract the original text corresponding to the given claim from the original response. When presented with an original response and a claim, reply with the original text. Make sure that your response is exactly the same as the original text, has no additional information beyond the given claim, and is enclosed in \\boxed{}."
    fullinput = instruction + "\n\nOriginal response: " + oneresponse + "\nClaim: " + curfact
    try:
        responsefromgpt = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content":fullinput}],
            temperature=0,
            stop=None
        )
    except Exception as exc:
        print(exc)
        return "gpt4o calling error"
    time.sleep(1.0)
    extractedtext = responsefromgpt.choices[0].message.content.strip("\\boxed{").strip("}")
    # print(extractedtext)
    if extractedtext in oneresponse:
        return extractedtext
    else:
        return "extracted text is not the orginal one"

def callgpt4oforentity(client, oneresponse, curtext):
    
    #####figure out the original sentence / atomic claim of the given fact
    # print("original response: ", oneresponse)
    # print("decomposed claim:", curtext)
    instruction = "Your task is to extract all entities contained in the given claim. When presented with a claim, reply with all the entities. Make sure that your response is exactly the same as the original entities, split by commas, and enclosed in a single \\boxed{}."
    fullinput = instruction + "\n\nClaim: " + curtext
    # print(fullinput)
    try:
        responsefromgpt = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content":fullinput}],
            temperature=0,
            stop=None
        )
    except Exception as exc:
        print(exc)
        return "gpt4o calling error"
    time.sleep(1.0)
    extractedentities = responsefromgpt.choices[0].message.content.strip("\\boxed{").strip("}")
    # print(extractedentities)
    allentities = extractedentities.split(",")
    # print(allentities)
    alltouseentities = []
    for oneentity in allentities:
        if oneentity.strip(" ") in oneresponse:
            alltouseentities.append(oneentity.strip(" "))
    # print(alltouseentities)
    # print(",".join(alltouseentities))
    if len(alltouseentities) != 0:
        return ",".join(alltouseentities)
    else:
        return "extracted entities are not the orginal ones"
    
def getorginaltext(allfacts, client, inputfile_withoriginaltext):
    newfile_withoriginaltext = inputfile_withoriginaltext
    
    if os.path.exists(newfile_withoriginaltext):
        print("Read existing original text file")
        with open(newfile_withoriginaltext,'r') as f:
            allfacts = json.load(f)
    
    keyindex = 0
    for onekey in allfacts.keys():
        print(keyindex)
        oneprompt = onekey.split("[/INST]")[0].split("[INST]")[1].strip()
        oneresponse = onekey.split("[/INST]")[1].strip()
        
        allsupported = allfacts[onekey]["supported"]
        allcontradicted = allfacts[onekey]["contradicted"]

        if len(allsupported) > 0:
            if "supported_original_flag" not in allfacts[onekey].keys():
                allfacts[onekey]["supported_original_flag"] = [0 for _ in range(len(allsupported))]
                allfacts[onekey]["supported_original_text"] = []
                for idx in range(len(allsupported)):
                    curfact = allsupported[idx]
                    originaltext = callgpt4o(client, oneresponse, curfact)
                    # print(originaltext)
                    allfacts[onekey]["supported_original_text"].append(originaltext)
                    if originaltext != "gpt4o calling error" and originaltext != "extracted text is not the orginal one":
                        allfacts[onekey]["supported_original_flag"][idx] = 1
            else:
                assert len(allfacts[onekey]["supported_original_flag"]) == len(allsupported)
                for idx in range(len(allsupported)):
                    if allfacts[onekey]["supported_original_flag"][idx] == 1:
                        continue
                    curfact = allsupported[idx]
                    originaltext = callgpt4o(client, oneresponse, curfact)
                    allfacts[onekey]["supported_original_text"][idx] = originaltext
                    if originaltext != "gpt4o calling error" and originaltext != "extracted text is not the orginal one":
                        allfacts[onekey]["supported_original_flag"][idx] = 1
            with open(newfile_withoriginaltext, "w") as f:
                json.dump(allfacts, f, indent=4, separators=(",", ": "))
            

        if len(allcontradicted) > 0:
            if "contradicted_original_flag" not in allfacts[onekey].keys():
                allfacts[onekey]["contradicted_original_flag"] = [0 for _ in range(len(allcontradicted))]
                allfacts[onekey]["contradicted_original_text"] = []
                for idx in range(len(allcontradicted)):
                    curfact = allcontradicted[idx]
                    originaltext = callgpt4o(client, oneresponse, curfact)
                    # print(originaltext)
                    allfacts[onekey]["contradicted_original_text"].append(originaltext)
                    if originaltext != "gpt4o calling error" and originaltext != "extracted text is not the orginal one":
                        allfacts[onekey]["contradicted_original_flag"][idx] = 1
                
            else:
                assert len(allfacts[onekey]["contradicted_original_flag"]) == len(allcontradicted)
                for idx in range(len(allcontradicted)):
                    if allfacts[onekey]["contradicted_original_flag"][idx] == 1:
                        continue
                    curfact = allcontradicted[idx]
                    originaltext = callgpt4o(client, oneresponse, curfact)
                    allfacts[onekey]["contradicted_original_text"][idx] = originaltext
                    if originaltext != "gpt4o calling error" and originaltext != "extracted text is not the orginal one":
                        allfacts[onekey]["contradicted_original_flag"][idx] = 1
            with open(newfile_withoriginaltext, "w") as f:
                json.dump(allfacts, f, indent=4, separators=(",", ": "))
        keyindex += 1


def getentities(allfacts, client, inputfile_withoriginaltext_entities):
    
    if os.path.exists(inputfile_withoriginaltext_entities):
        print("Read existing original text and entity file")
        with open(inputfile_withoriginaltext_entities,'r') as f:
            allfacts = json.load(f)
    
    keyindex = 0
    for onekey in allfacts.keys():
        print(keyindex)
        oneprompt = onekey.split("[/INST]")[0].split("[INST]")[1].strip()
        oneresponse = onekey.split("[/INST]")[1].strip()
        
        allsupported = allfacts[onekey]["supported"]
        allcontradicted = allfacts[onekey]["contradicted"]

        if len(allsupported) > 0 and "supported_original_flag" in allfacts[onekey].keys():
            if "supported_entity_flag" not in allfacts[onekey].keys():
                allfacts[onekey]["supported_entity_flag"] = [0 for _ in range(len(allsupported))]
                allfacts[onekey]["supported_entity"] = []
                for idx in range(len(allsupported)):
                    curoriginaltext = allfacts[onekey]["supported_original_text"][idx]
                    curoriginalflag = allfacts[onekey]["supported_original_flag"][idx]
                    if curoriginalflag == 1:
                        allentities = callgpt4oforentity(client, oneresponse, curoriginaltext)
                    else:
                        allentities = "no original text"
                    # print(allentities)
                    # exit -1
                    allfacts[onekey]["supported_entity"].append(allentities)
                    if allentities != "gpt4o calling error" and allentities != "extracted entities are not the orginal ones" and allentities != "no original text":
                        allfacts[onekey]["supported_entity_flag"][idx] = 1
            else:
                assert len(allfacts[onekey]["supported_entity_flag"]) == len(allsupported)
                for idx in range(len(allsupported)):
                    if allfacts[onekey]["supported_entity_flag"][idx] == 1:
                        continue
                    curoriginaltext = allfacts[onekey]["supported_original_text"][idx]
                    curoriginalflag = allfacts[onekey]["supported_original_flag"][idx]
                    if curoriginalflag == 1:
                        allentities = callgpt4oforentity(client, oneresponse, curoriginaltext)
                    else:
                        allentities = "no original text"
                    allfacts[onekey]["supported_entity"][idx] = allentities
                    if allentities != "gpt4o calling error" and allentities != "extracted entities are not the orginal ones" and allentities != "no original text":
                        allfacts[onekey]["supported_entity_flag"][idx] = 1
            with open(inputfile_withoriginaltext_entities, "w") as f:
                json.dump(allfacts, f, indent=4, separators=(",", ": "))

        if len(allcontradicted) > 0 and "contradicted_original_flag" in allfacts[onekey].keys():
            if "contradicted_entity_flag" not in allfacts[onekey].keys():
                allfacts[onekey]["contradicted_entity_flag"] = [0 for _ in range(len(allcontradicted))]
                allfacts[onekey]["contradicted_entity"] = []
                for idx in range(len(allcontradicted)):
                    curoriginaltext = allfacts[onekey]["contradicted_original_text"][idx]
                    curoriginalflag = allfacts[onekey]["contradicted_original_flag"][idx]
                    if curoriginalflag == 1:
                        allentities = callgpt4oforentity(client, oneresponse, curoriginaltext)
                    else:
                        allentities = "no original text"
                    # print(allentities)
                    # exit -1
                    allfacts[onekey]["contradicted_entity"].append(allentities)
                    if allentities != "gpt4o calling error" and allentities != "extracted entities are not the orginal ones" and allentities != "no original text":
                        allfacts[onekey]["contradicted_entity_flag"][idx] = 1
            else:
                assert len(allfacts[onekey]["contradicted_entity_flag"]) == len(allcontradicted)
                for idx in range(len(allcontradicted)):
                    if allfacts[onekey]["contradicted_entity_flag"][idx] == 1:
                        continue
                    curoriginaltext = allfacts[onekey]["contradicted_original_text"][idx]
                    curoriginalflag = allfacts[onekey]["contradicted_original_flag"][idx]
                    if curoriginalflag == 1:
                        allentities = callgpt4oforentity(client, oneresponse, curoriginaltext)
                    else:
                        allentities = "no original text"
                    allfacts[onekey]["contradicted_entity"][idx] = allentities
                    if allentities != "gpt4o calling error" and allentities != "extracted entities are not the orginal ones" and allentities != "no original text":
                        allfacts[onekey]["contradicted_entity_flag"][idx] = 1
            with open(inputfile_withoriginaltext_entities, "w") as f:
                json.dump(allfacts, f, indent=4, separators=(",", ": "))
        keyindex += 1


def handle_decomposed_facts():
    
    set_seed(42)

    # temperature = 1.0
    temperature = 0
    top_p = 0.9
    tocheckmodel = "llama3-8b-instruct"

    if temperature == 0:
        tempath = "longfact-allfacts"
    else:
        tempath = "longfact-allfacts_"+ tocheckmodel + "_temperature_" + str(temperature) + "_topp_" + str(top_p)
    
    # inputfile = "./longfact-ACE-decomposed-facts/longfact-allfacts.json"
    inputfile = "./longfact-ACE-decomposed-facts/" + tempath + ".json"
    allfacts = {}
    with open(inputfile,'r') as f:
        allfacts = json.load(f)
    print(len(allfacts))

    api_key = "OPENAI_KEY"
    client = OpenAI(api_key=api_key)

    # inputfile_withoriginaltext = "./longfact-ACE-decomposed-facts/longfact-allfacts-withoriginaltext.json"
    inputfile_withoriginaltext = "./longfact-ACE-decomposed-facts/" + tempath + "-withoriginaltext.json"

    #####get original text, no need for this file (as this file is only used to obtain entities)
    # getorginaltext(allfacts, client, inputfile_withoriginaltext)
    # exit -1

    # print("-------------------------------------------------------------------------------")
    
    with open(inputfile_withoriginaltext,'r') as f:
        allfacts = json.load(f)
    print(len(allfacts))

    inputfile_withoriginaltext_entities = "./longfact-ACE-decomposed-facts/" + tempath + "-withoriginaltext-entities.json"

    # ###get entities
    # getentities(allfacts, client, inputfile_withoriginaltext_entities)
    # exit -1

    with open(inputfile_withoriginaltext_entities,'r') as f:
        allfacts = json.load(f)
    print(len(allfacts))
    
    numofsupported = 0
    numofnotsupported = 0
    numofcontradicted = 0
    numofnoreference = 0
    numofsearcherror = 0
    for onekey in allfacts.keys():
        numofsupported += len(allfacts[onekey]["supported"])
        numofnotsupported += len(allfacts[onekey]["not_supported"])
        numofcontradicted += len(allfacts[onekey]["contradicted"])
        numofnoreference += len(allfacts[onekey]["no_reference"])
        numofsearcherror += len(allfacts[onekey]["search_error"])
    print("number of supported: ", numofsupported)
    print("number of not supported: ", numofnotsupported)
    print("number of contradicted: ", numofcontradicted)
    print("number of no reference: ", numofnoreference)
    print("number of search error: ", numofsearcherror)
    # exit -1

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    model.double()             ####for getting same probability between generate and forward
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()

    allsize = 0
    allsupportedres = []
    allcontradictedres = []
    for onekey in allfacts.keys():
        oneprompt = onekey.split("[/INST]")[0].split("[INST]")[1].strip()
        oneresponse = onekey.split("[/INST]")[1].strip()
        input_tokens_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": oneprompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        # print(input_tokens_chat)
        input_ids_chat = tokenizer(input_tokens_chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        # print(input_ids_chat.shape)
        # print(input_ids_chat)
        allsupported = allfacts[onekey]["supported"]
        allcontradicted = allfacts[onekey]["contradicted"]

        # ####get probability
        # print("all supported")
        # if len(allsupported) > 0:
        #     for idx in range(len(allsupported)):
        #         onesupported = allsupported[idx]
        #         handleonefact(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, onesupported)

        # print("all contradicted")
        # if len(allcontradicted) > 0:
        #     for idx in range(len(allcontradicted)):
        #         onecontradicted = allcontradicted[idx]
        #         handleonefact(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, onecontradicted)       
        if len(allsupported) > 0:
            allsupported_original_flag = allfacts[onekey]["supported_original_flag"]
            allsupported_original_text = allfacts[onekey]["supported_original_text"]
            allsupported_entity_flag = allfacts[onekey]["supported_entity_flag"]
            allsupported_entity = allfacts[onekey]["supported_entity"]
            assert len(allsupported) == len(allsupported_original_flag) == len(allsupported_original_text) == len(allsupported_entity_flag) == len(allsupported_entity)
            allsupported_flag_nozero_idx = []
            for idx in range(len(allsupported_original_flag)):
                if allsupported_original_flag[idx] == 1 and allsupported_entity_flag[idx] == 1:
                    allsupported_flag_nozero_idx.append(idx)
        
        if len(allcontradicted) > 0:
            allcontradicted_original_flag = allfacts[onekey]["contradicted_original_flag"]
            allcontradicted_original_text = allfacts[onekey]["contradicted_original_text"]
            allcontradicted_entity_flag = allfacts[onekey]["contradicted_entity_flag"]
            allcontradicted_entity = allfacts[onekey]["contradicted_entity"]
            assert len(allcontradicted) == len(allcontradicted_original_flag) == len(allcontradicted_original_text) == len(allcontradicted_entity_flag) == len(allcontradicted_entity)
            allcontradicted_flag_nozero_idx = []
            for idx in range(len(allcontradicted_original_flag)):
                if allcontradicted_original_flag[idx] == 1 and allcontradicted_entity_flag[idx] == 1:
                    allcontradicted_flag_nozero_idx.append(idx)
        
        if len(allsupported) > 0 and len(allcontradicted) > 0:
            minsize = min(len(allsupported_flag_nozero_idx), len(allcontradicted_flag_nozero_idx))
            if minsize > 0:
                sampled_supported_idx = random.sample(allsupported_flag_nozero_idx, minsize)
                sampled_contradicted_idx = random.sample(allcontradicted_flag_nozero_idx, minsize)
                allsize += minsize
                for idx in range(minsize):
                    one_supported_idx = sampled_supported_idx[idx]
                    onesupported = allsupported[one_supported_idx]
                    onesupported_original_text = allsupported_original_text[one_supported_idx]
                    onesupported_entity = allsupported_entity[one_supported_idx]
                    res_prob_s = getprobabilityentity(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, onesupported, onesupported_original_text, onesupported_entity)
                    allsupportedres.append(res_prob_s)

                    one_contradicted_idx = sampled_contradicted_idx[idx]
                    onecontradicted = allcontradicted[one_contradicted_idx]
                    onecontradicted_original_text = allcontradicted_original_text[one_contradicted_idx]
                    onecontradicted_entity = allcontradicted_entity[one_contradicted_idx]
                    res_prob_c = getprobabilityentity(model, tokenizer, oneresponse, input_tokens_chat, input_ids_chat, onecontradicted, onecontradicted_original_text, onecontradicted_entity)
                    allcontradictedres.append(res_prob_c)                
    
    print(allsize)
    # idx2name = {
    #     0: "arithmetic_mean",
    #     1: "geometric_mean",
    #     2: "min_prob",
    #     3: "min_3_prob",
    #     4: "min_5_prob",
    #     5: "min_5percent_prob",
    #     6: "min_10percent_prob",
    #     7: "min_15percent_prob"
    # }
    idx2name = {
        0: "arithmetic_average",
        1: "geometric_average",
        2: "lowest_1",
        3: "lowest_3",
        4: "lowest_5",
        5: "lowest_5%",
        6: "lowest_10%",
        7: "lowest_15%"
    }
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    for i in range(len(allsupportedres[0])):
        tousesupportedres = [allsupportedres[idx][i] for idx in range(len(allsupportedres))]
        assert len(tousesupportedres) == allsize
        tousecontradictedres = [allcontradictedres[idx][i] for idx in range(len(allcontradictedres))]
        assert len(tousecontradictedres) == allsize
        tousesupportedres.sort()
        tousecontradictedres.sort()
        print(idx2name[i])
        print(format(sum(tousesupportedres) / len(tousesupportedres), '.3f'))
        print(format(sum(tousecontradictedres) / len(tousecontradictedres), '.3f'))
        diff = sum(tousesupportedres) / len(tousesupportedres) - sum(tousecontradictedres) / len(tousecontradictedres)
        print(format(diff, '.3f'))
        print("-------------------------------------------------------------------------------------------------------")

        bins = np.linspace(0, 1, 21)
        
        data_bucketed_s = np.digitize(tousesupportedres, bins)
        allfigres_s = [0 for _ in range(len(bins))]
        for onedata in data_bucketed_s:
            allfigres_s[onedata] += 1
        max_allfigres_s = max(allfigres_s)
        # axs[i//4][i%4].bar(bins, allfigres_s, color='blue', alpha=0.5, label='Correct Claim', width=0.05)
        axs[i//4][i%4].bar(bins, allfigres_s, color='blue', alpha=0.5, label='Factual', width=0.05)

        data_bucketed_c = np.digitize(tousecontradictedres, bins)
        allfigres_c = [0 for _ in range(len(bins))]
        for onedata in data_bucketed_c:
            allfigres_c[onedata] += 1
        max_allfigres_c = max(allfigres_c)
        max_y = max(max_allfigres_s, max_allfigres_c)
        # print(max_y)
        # axs[i//4][i%4].bar(bins, allfigres_c, color='orange', alpha=0.5, label='Wrong Claim', width=0.05)
        axs[i//4][i%4].bar(bins, allfigres_c, color='orange', alpha=0.5, label='Hallucinated', width=0.05)

        axs[i//4][i%4].set_xticks(np.arange(0, 1, 0.2))
        # axs[i//4][i%4].set_yticks(np.arange(0, max_y + 5, 5))
        if i == 0 or i == 1:
            axs[i//4][i%4].set_yticks(np.arange(0, max_y + 5, 10))
        else:
            axs[i//4][i%4].set_yticks(np.arange(0, max_y + 5, 5))

        axs[i//4][i%4].set_xlabel(idx2name[i], fontsize=20)
        axs[i//4][i%4].set_ylabel("Counts", fontsize=20)
    axs[0, 0].legend(fontsize=12)
    if temperature == 0:
        pictitle = "Greedy Decoding"
    else:
        pictitle = "Temperature " + str(temperature)
    # plt.suptitle(pictitle, fontsize=32)
    # picname = "Figures/entities_temperature_" + str(temperature) + "_newlegend.png"
    picname = "Figures/entities_temperature_" + str(temperature) + "_newlegend.pdf"
    plt.savefig(picname)
    print(allsize)
    
handle_decomposed_facts()