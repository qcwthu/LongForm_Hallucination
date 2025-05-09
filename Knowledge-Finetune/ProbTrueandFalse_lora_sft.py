from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()
import torch
import torch.nn.functional as F
import json
import os
import random
import numpy as np
import time
from typing import List
import matplotlib.pyplot as plt
from openai import OpenAI
import heapq
from sklearn.metrics import roc_auc_score
# from vllm import LLM, SamplingParams
import argparse
import csv
import pandas as pd
from openai import OpenAI
import gc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def getTrueorFalse(model, tokenizer, question_text_original, terminators):
    question_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_text_original}],
        add_generation_prompt=True,
        tokenize=False
    )
    question_text_ids = tokenizer(question_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    question_text_ids_true_3 = tokenizer(question_text + " True", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    question_text_ids_true_4 = tokenizer(question_text + " true", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    question_text_ids_false_3 = tokenizer(question_text + " False", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    question_text_ids_false_4 = tokenizer(question_text + " false", return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    outputs = model.generate(
        question_text_ids,
        max_new_tokens=1,
        eos_token_id=terminators,
        do_sample=False,
        output_attentions=True,
        output_logits=True,
        return_dict_in_generate=True
    )
    newoutput = outputs.sequences[0][question_text_ids.shape[-1]:]
    # print(newoutput[0])
    decoded_response = tokenizer.decode(newoutput[0])
    # print(decoded_response)
    # exit -1
    outputlogits = outputs.logits[0][0:1]
    outputlogits_sm = torch.nn.functional.softmax(outputlogits, dim=-1)
    # print(outputlogits_sm.shape)

    True_1 = tokenizer.convert_tokens_to_ids(["True"])
    True_2 = tokenizer.convert_tokens_to_ids(["true"])
    True_3 = question_text_ids_true_3[0][-1:]
    True_4 = question_text_ids_true_4[0][-1:]
    # print(True_1, True_2, True_3, True_4)
    False_1 = tokenizer.convert_tokens_to_ids(["False"])
    False_2 = tokenizer.convert_tokens_to_ids(["false"])
    False_3 = question_text_ids_false_3[0][-1:]
    False_4 = question_text_ids_false_4[0][-1:]
    # print(False_1, False_2, False_3, False_4)
    
    allprobs = []
    oneprobforTrue_1 = outputlogits_sm[0][True_1[0]].detach().cpu().item()
    oneprobforTrue_2 = outputlogits_sm[0][True_2[0]].detach().cpu().item()
    oneprobforTrue_3 = outputlogits_sm[0][True_3[0]].detach().cpu().item()
    oneprobforTrue_4 = outputlogits_sm[0][True_4[0]].detach().cpu().item()
    
    oneprobforFalse_1 = outputlogits_sm[0][False_1[0]].detach().cpu().item()
    oneprobforFalse_2 = outputlogits_sm[0][False_2[0]].detach().cpu().item()
    oneprobforFalse_3 = outputlogits_sm[0][False_3[0]].detach().cpu().item()
    oneprobforFalse_4 = outputlogits_sm[0][False_4[0]].detach().cpu().item()

    # print(oneprobforTrue_1, oneprobforTrue_2, oneprobforTrue_3, oneprobforTrue_4)
    # print(oneprobforFalse_1, oneprobforFalse_2, oneprobforFalse_3, oneprobforFalse_4)
    allprobs.append(oneprobforTrue_1 + oneprobforTrue_2 + oneprobforTrue_3 + oneprobforTrue_4)
    allprobs.append(oneprobforFalse_1 + oneprobforFalse_2 + oneprobforFalse_3 + oneprobforFalse_4)
    return allprobs, decoded_response

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

def compute_auroc(labels, probs, multi_class="ovr", **kwargs):
    one_hot_labels = (
        F.one_hot(labels, num_classes=probs.size(-1)) if labels.ndim == 1 else labels
    )

    try:
        auroc = roc_auc_score(one_hot_labels, probs.float(), multi_class=multi_class, **kwargs)
    except ValueError:
        auroc = float("nan")
        logging.exception("AUROC calculation failed.", exc_info=True)

    return auroc

def plot_res(all_support_result, all_contradicted_result):
    idx2name = {
        0: "P(Factual)",
    }
    alltoplot = [[all_support_result, all_contradicted_result]]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
    tousesupportedres = alltoplot[0][0]
    tousecontradictedres = alltoplot[0][1]
    tousesupportedres.sort()
    tousecontradictedres.sort()

    bins = np.linspace(0, 1, 21)
    
    data_bucketed_s = np.digitize(tousesupportedres, bins)
    allfigres_s = [0 for _ in range(len(bins))]
    for onedata in data_bucketed_s:
        allfigres_s[onedata] += 1
    max_allfigres_s = max(allfigres_s)
    axs.bar(bins, allfigres_s, color='blue', alpha=0.5, label='Factual', width=0.05)

    data_bucketed_c = np.digitize(tousecontradictedres, bins)
    allfigres_c = [0 for _ in range(len(bins))]
    for onedata in data_bucketed_c:
        allfigres_c[onedata] += 1
    max_allfigres_c = max(allfigres_c)
    max_y = max(max_allfigres_s, max_allfigres_c)
    # print(max_y)
    axs.bar(bins, allfigres_c, color='orange', alpha=0.5, label='Hallucinated', width=0.05)

    axs.set_xticks(np.arange(0, 1, 0.2))
    axs.set_yticks(np.arange(0, max_y + 5, 20))

    axs.set_xlabel(idx2name[0], fontsize=20)
    axs.set_ylabel("Counts", fontsize=20)

    axs.legend(fontsize=16)
    pictitle = "SFT Model"
    plt.suptitle(pictitle, fontsize=32)
    picname = "Figures/True_False_sft_model" + ".png"
    plt.savefig(picname)
    picname_1 = "Figures/True_False_sft_model" + ".pdf"
    plt.savefig(picname_1)

def handle_decomposed_facts():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
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
    tousetraindata = train_data + valid_data
    tousetestdata = test_data
    # tousetestdata = valid_data
    print(len(tousetraindata), len(tousetestdata))
    all_supported_facts_train = []
    all_contradicted_facts_train = []
    for onesample in tousetraindata:
        cur_q = onesample["question"].replace("\nResponse: ", "").replace("\n\n", "\n")         #### make it the same as the template in Llama-factory
        if onesample["label"] == 1:
            all_contradicted_facts_train.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_train.append(cur_q)
        else:
            print("train label error!")

    all_supported_facts_test = []
    all_contradicted_facts_test = []
    for onesample in tousetestdata:
        cur_q = onesample["question"].replace("\nResponse: ", "").replace("\n\n", "\n")
        if onesample["label"] == 1:
            all_contradicted_facts_test.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_test.append(cur_q)
        else:
            print("test label error!")
    print(len(all_supported_facts_train), len(all_contradicted_facts_train), len(all_supported_facts_test), len(all_contradicted_facts_test))
    
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #### lora sft
    model_id = "/data/home/LLaMA-Factory/models/llama3_8b_automized_claim_classifer_lora_sft_target_all"

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

    all_labels = []
    all_scores = []
    all_score_pos = []
    all_support_result = []
    all_contradicted_result = []
    all_support_query = []
    all_contradicted_query = []
    allaccforq1_support_token = 0
    allaccforq1_contradicted_token = 0
    # for onesupported_question_text_1 in all_supported_facts_train:
    for onesupported_question_text_1 in all_supported_facts_test:
        onedata = onesupported_question_text_1.strip().split("\n")
        assert len(onedata) == 2
        query = onedata[1]
        onlyquery = query.replace("Claim: ", "")
        all_support_query.append(onlyquery)
        support_res1, support_response_q1 = getTrueorFalse(model, tokenizer, onesupported_question_text_1, terminators)
        oneres = support_res1[0] / (support_res1[0] + support_res1[1])      ####seems to be equal to the difference between support_res1[0] and support_res1[1]
        all_support_result.append(oneres)
        all_labels.append(0)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if support_response_q1.strip().lower() == 'true':
            allaccforq1_support_token += 1
        
    # for onecontradicted_question_text_1 in all_contradicted_facts_train:
    for onecontradicted_question_text_1 in all_contradicted_facts_test:
        onedata = onecontradicted_question_text_1.strip().split("\n")
        assert len(onedata) == 2
        query = onedata[1]
        onlyquery = query.replace("Claim: ", "")
        all_contradicted_query.append(onlyquery)
        contradicted_res1, contradicted_response_q1 = getTrueorFalse(model, tokenizer, onecontradicted_question_text_1, terminators)
        oneres = contradicted_res1[0] / (contradicted_res1[0] + contradicted_res1[1])
        all_contradicted_result.append(oneres)
        all_labels.append(1)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if contradicted_response_q1.strip().lower() == 'false':
            allaccforq1_contradicted_token += 1
    
    all_label_tensor = torch.tensor(all_labels)
    all_score_tensor = torch.tensor(all_scores)
    auroc = compute_auroc(all_label_tensor, all_score_tensor)
    auc_sklearn = roc_auc_score(all_labels, all_score_pos)
    print("auroc: ", auroc, " auc_sklearn: ", auc_sklearn)
    print(len(all_supported_facts_test), allaccforq1_support_token, len(all_contradicted_facts_test), allaccforq1_contradicted_token)
    
    correct_acc_direct = allaccforq1_support_token / len(all_supported_facts_test)
    wrong_acc_direct = allaccforq1_contradicted_token / len(all_contradicted_facts_test)
    balance_acc_direct = (correct_acc_direct + wrong_acc_direct) / 2
    print("correct accuracy direct: ", correct_acc_direct)
    print("wrong accurayc direct:", wrong_acc_direct)
    print("balanced accuracy direct: ", balance_acc_direct)

    #### plot
    #### ******************************************************
    # plot_res(all_support_result, all_contradicted_result)
    #### ******************************************************

    # all_thresholds = [0.05 * i for i in range(0, 20)]
    # all_thresholds = [0.05 * i for i in range(10, 20)]
    # all_thresholds.extend([0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999])
    all_thresholds = [0.02 * i for i in range(15, 36)]
    print(all_thresholds)
    
    best_acc = -1
    best_threshold = -1.0
    threshold_for_case_study = 0.44
    all_support_answer = []
    all_contradicted_answer = []
    for threshold in all_thresholds:
        print("*********************************************************************************************************************")
        print("threshold: ", threshold)
        supportfact_predict_as_true = 0
        supportfact_predict_as_false = 0
        contradictedfact_predict_as_true = 0
        contradictedfact_predict_as_false = 0
        for oneres in all_support_result:
            if oneres >= threshold:
                supportfact_predict_as_true += 1
                if threshold == threshold_for_case_study:
                    all_support_answer.append("true")
            else:
                supportfact_predict_as_false += 1
                if threshold == threshold_for_case_study:
                    all_support_answer.append("false")
        for oneres in all_contradicted_result:
            if oneres < threshold:
                contradictedfact_predict_as_false += 1
                if threshold == threshold_for_case_study:
                    all_contradicted_answer.append("false")
            else:
                contradictedfact_predict_as_true += 1
                if threshold == threshold_for_case_study:
                    all_contradicted_answer.append("true")

        print("P(True)/(P(True)+P(False)) support accuracy: ", supportfact_predict_as_true, len(all_support_result), supportfact_predict_as_true / len(all_support_result))
        print("P(True)/(P(True)+P(False)) contradicted accuracy: ", contradictedfact_predict_as_false, len(all_contradicted_result), contradictedfact_predict_as_false / len(all_contradicted_result))
        
        print(supportfact_predict_as_true, supportfact_predict_as_false, contradictedfact_predict_as_false, contradictedfact_predict_as_true)
        precision = contradictedfact_predict_as_false / (contradictedfact_predict_as_false + supportfact_predict_as_false) if (contradictedfact_predict_as_false + supportfact_predict_as_false) > 0 else 0
        recall = contradictedfact_predict_as_false / len(all_contradicted_result)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        print("precision: ", format(precision, '.3f'))
        print("recall: ", format(recall, '.3f'))
        print("f1_score: ", format(f1_score, '.3f'))

        correct_acc = supportfact_predict_as_true / len(all_support_result)
        wrong_acc = contradictedfact_predict_as_false / len(all_contradicted_result)
        balance_acc = (correct_acc + wrong_acc) / 2
        print("correct accuracy: ", format(correct_acc, '.3f'))
        print("wrong accurayc:", format(wrong_acc, '.3f'))
        print("balanced accuracy: ", format(balance_acc, '.3f'))
        if balance_acc > best_acc:
            best_acc = balance_acc
            best_threshold = threshold
        print("*********************************************************************************************************************")
    print("best_acc: ", best_acc)
    print("best_threshold: ", best_threshold)
    print(len(all_support_query), len(all_support_answer))
    print(len(all_contradicted_query), len(all_contradicted_answer))

def handle_decomposed_facts_cot_true_false():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
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
    tousetraindata = train_data + valid_data
    tousetestdata = test_data
    # tousetestdata = valid_data
    print(len(tousetraindata), len(tousetestdata))
    all_supported_facts_train = []
    all_contradicted_facts_train = []
    instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first explain the solution and then enclose the ultimate answer ('True' or 'False') in \\boxed{{}}."
    for onesample in tousetraindata:
        q_text = onesample["question"].split("\n\n")[-1].replace("\nResponse: ", "")
        cur_q = instruction + "\n" + q_text
        if onesample["label"] == 1:
            all_contradicted_facts_train.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_train.append(cur_q)
        else:
            print("train label error!")

    all_supported_facts_test = []
    all_contradicted_facts_test = []
    for onesample in tousetestdata:
        q_text = onesample["question"].split("\n\n")[-1].replace("\nResponse: ", "")
        cur_q = instruction + "\n" + q_text
        if onesample["label"] == 1:
            all_contradicted_facts_test.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_test.append(cur_q)
        else:
            print("test label error!")
    print(len(all_supported_facts_train), len(all_contradicted_facts_train), len(all_supported_facts_test), len(all_contradicted_facts_test))
    
    tensor_parallel = 8
    model_id = "/data/home/LLaMA-Factory/models/llama3_8b_automized_claim_classifer_lora_sft_target_all_cot_true_false_diff_prompt_qa_format"

    llm = LLM(model_id, tensor_parallel_size=tensor_parallel)
    tokenizer = llm.get_tokenizer()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        stop_token_ids=terminators,
        use_beam_search=False
    )

    processed_supported_questions = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": oneq}],
            add_generation_prompt=True,
            tokenize=False
        # ) for oneq in all_supported_facts_train
        ) for oneq in all_supported_facts_test
    ]

    processed_contradicted_questions = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": oneq}],
            add_generation_prompt=True,
            tokenize=False
        # ) for oneq in all_contradicted_facts_train
        ) for oneq in all_contradicted_facts_test
    ]

    all_supported_res = {}
    all_supported_outputs = llm.generate(processed_supported_questions, sampling_params)
    supportfact_predict_as_true = 0
    supportfact_predict_as_false = 0
    for idx in range(len(all_supported_outputs)):
        # question = all_supported_facts_train[idx]
        question = all_supported_facts_test[idx]
        output = all_supported_outputs[idx]
        generated_text = output.outputs[0].text
        onekey = str(idx) + "@@@@@@" + question
        all_supported_res[onekey] = generated_text
        onepred = generated_text.split("\\boxed{")[-1].split("}")[0].lower()
        if onepred == "true":
            supportfact_predict_as_true += 1
        elif onepred == "false":
            supportfact_predict_as_false += 1
        else:
            onepred_exclude_text = onepred.split("\\text{")[-1].split("}")[0]
            if onepred_exclude_text == "true":
                supportfact_predict_as_true += 1
            elif onepred_exclude_text == "false":
                supportfact_predict_as_false += 1
            else:
                print("one output format error! ", onepred_exclude_text)
                supportfact_predict_as_false += 1       ### can change to random selecting from True and False
    
    all_contradicted_res = {}
    all_contradicted_outputs = llm.generate(processed_contradicted_questions, sampling_params)
    contradictedfact_predict_as_true = 0
    contradictedfact_predict_as_false = 0
    for idx in range(len(all_contradicted_outputs)):
        # question = all_contradicted_facts_train[idx]
        question = all_contradicted_facts_test[idx]
        output = all_contradicted_outputs[idx]
        generated_text = output.outputs[0].text
        onekey = str(idx) + "@@@@@@" + question
        all_contradicted_res[onekey] = generated_text
        onepred = generated_text.split("\\boxed{")[-1].split("}")[0].lower()
        if onepred == "true":
            contradictedfact_predict_as_true += 1
        elif onepred == "false":
            contradictedfact_predict_as_false += 1
        else:
            onepred_exclude_text = onepred.split("\\text{")[-1].split("}")[0]
            if onepred_exclude_text == "true":
                contradictedfact_predict_as_true += 1
            elif onepred_exclude_text == "false":
                contradictedfact_predict_as_false += 1
            else:
                # print("one output format error! ", onepred_exclude_text)
                contradictedfact_predict_as_true += 1
    
    allres = {
        "all_supported_res": all_supported_res,
        "all_contradicted_res": all_contradicted_res
    }
    savepath = "./CoTResult/ft_cot_true_false_diff_prompt_qa_format"
    with open(savepath, "w") as f:
        json.dump(allres, f, indent=4, separators=(",", ": "))

    print(supportfact_predict_as_true, supportfact_predict_as_false, len(processed_supported_questions))
    print(contradictedfact_predict_as_true, contradictedfact_predict_as_false, len(processed_contradicted_questions))
    correct_acc_direct = supportfact_predict_as_true / len(processed_supported_questions)
    wrong_acc_direct = contradictedfact_predict_as_false / len(processed_contradicted_questions)
    balance_acc_direct = (correct_acc_direct + wrong_acc_direct) / 2
    print("correct accuracy direct: ", correct_acc_direct)
    print("wrong accurayc direct:", wrong_acc_direct)
    print("balanced accuracy direct: ", balance_acc_direct)


def handle_decomposed_facts_cot_true_false_new():
    savepath = "./CoTResult/ft_cot_true_false_diff_prompt_qa_format"
    with open(savepath,'r') as f:
        allres = json.load(f)
    all_supported_res = allres["all_supported_res"]
    all_contradicted_res = allres["all_contradicted_res"]
    
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

    all_labels = []
    all_scores = []
    all_score_pos = []
    all_support_result = []
    all_contradicted_result = []
    allaccforq1_support_token = 0
    allaccforq1_contradicted_token = 0
    new_instruction = f"Your task is to determine the correctness of the given claim based on the corresponding solution. When presented with a claim and the solution, reply with 'True' or 'False'. Make sure that your response is exactly 'True' or 'False' without any extra commentary whatsoever."
    for onekey in all_supported_res.keys():
        oneinput = onekey.split("@@@@@@")[-1].split("Claim: ")[-1]
        oneresponse = all_supported_res[onekey]
        onetouseinput = new_instruction + "\n\nClaim: " + oneinput + "\nSolution: " + oneresponse + "\nResponse: "
        support_res1, support_response_q1 = getTrueorFalse(model, tokenizer, onetouseinput, terminators)
        # print(support_response_q1)
        oneres = support_res1[0] / (support_res1[0] + support_res1[1])      ####seems to be equal to the difference between support_res1[0] and support_res1[1]
        all_support_result.append(oneres)
        all_labels.append(0)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if support_response_q1.strip().lower() == 'true':
            allaccforq1_support_token += 1
    
    for onekey in all_contradicted_res.keys():
        oneinput = onekey.split("@@@@@@")[-1].split("Claim: ")[-1]
        oneresponse = all_contradicted_res[onekey]
        onetouseinput = new_instruction + "\n\nClaim: " + oneinput + "\nSolution: " + oneresponse + "\nResponse: "
        contradicted_res1, contradicted_response_q1 = getTrueorFalse(model, tokenizer, onetouseinput, terminators)
        oneres = contradicted_res1[0] / (contradicted_res1[0] + contradicted_res1[1])
        all_contradicted_result.append(oneres)
        all_labels.append(1)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if contradicted_response_q1.strip().lower() == 'false':
            allaccforq1_contradicted_token += 1    
    
    print(len(all_supported_res), allaccforq1_support_token, len(all_contradicted_res), allaccforq1_contradicted_token)
    precision_direct = allaccforq1_contradicted_token / (allaccforq1_contradicted_token + len(all_supported_res) - allaccforq1_support_token)
    recall_direct = allaccforq1_contradicted_token / len(all_contradicted_res)
    f1_direct = 2 * precision_direct * recall_direct / (precision_direct + recall_direct)
    print("precison: ", precision_direct)
    print("recall: ", recall_direct)
    print("f1_direct: ", f1_direct)

    correct_acc_direct = allaccforq1_support_token / len(all_supported_res)
    wrong_acc_direct = allaccforq1_contradicted_token / len(all_contradicted_res)
    balance_acc_direct = (correct_acc_direct + wrong_acc_direct) / 2
    print("correct accuracy direct: ", correct_acc_direct)
    print("wrong accurayc direct:", wrong_acc_direct)
    print("balanced accuracy direct: ", balance_acc_direct)
    

    all_label_tensor = torch.tensor(all_labels)
    all_score_tensor = torch.tensor(all_scores)
    auroc = compute_auroc(all_label_tensor, all_score_tensor)
    auc_sklearn = roc_auc_score(all_labels, all_score_pos)
    print("auroc: ", auroc, " auc_sklearn: ", auc_sklearn)

    all_thresholds = [0.02 * i for i in range(15, 36)]
    print(all_thresholds)
    
    best_acc = -1
    best_threshold = -1.0
    for threshold in all_thresholds:
        print("*********************************************************************************************************************")
        print("threshold: ", threshold)
        supportfact_predict_as_true = 0
        supportfact_predict_as_false = 0
        contradictedfact_predict_as_true = 0
        contradictedfact_predict_as_false = 0
        for oneres in all_support_result:
            if oneres >= threshold:
                supportfact_predict_as_true += 1
            else:
                supportfact_predict_as_false += 1
        for oneres in all_contradicted_result:
            if oneres < threshold:
                contradictedfact_predict_as_false += 1
            else:
                contradictedfact_predict_as_true += 1

        precision = contradictedfact_predict_as_false / (contradictedfact_predict_as_false + supportfact_predict_as_false) if (contradictedfact_predict_as_false + supportfact_predict_as_false) > 0 else 0
        recall = contradictedfact_predict_as_false / len(all_contradicted_result)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        print("precision: ", format(precision, '.3f'))
        print("recall: ", format(recall, '.3f'))
        print("f1_score: ", format(f1_score, '.3f'))

        correct_acc = supportfact_predict_as_true / len(all_support_result)
        wrong_acc = contradictedfact_predict_as_false / len(all_contradicted_result)
        balance_acc = (correct_acc + wrong_acc) / 2
        print("correct accuracy: ", format(correct_acc, '.3f'))
        print("wrong accurayc:", format(wrong_acc, '.3f'))
        print("balanced accuracy: ", format(balance_acc, '.3f'))
        if balance_acc > best_acc:
            best_acc = balance_acc
            best_threshold = threshold
        print("*********************************************************************************************************************")
    print("best_acc: ", best_acc)
    print("best_threshold: ", best_threshold)


def handle_decomposed_facts_true_false_cot():
    
    set_seed(42)
    train_split = 0.7
    valid_split = 0.1
    data_path = "./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy.json"
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
    tousetraindata = train_data + valid_data
    tousetestdata = test_data
    # tousetestdata = valid_data
    print(len(tousetraindata), len(tousetestdata))
    all_supported_facts_train = []
    all_contradicted_facts_train = []
    instruction = f"Your task is to determine the correctness of the given claim. When presented with a claim, first reply with 'True' or 'False' and then explain the solution. Make sure that your response starts with 'True' or 'False'."
    for onesample in tousetraindata:
        q_text = onesample["question"].split("\n\n")[-1].replace("\nResponse: ", "")
        cur_q = instruction + "\n" + q_text
        if onesample["label"] == 1:
            all_contradicted_facts_train.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_train.append(cur_q)
        else:
            print("train label error!")

    all_supported_facts_test = []
    all_contradicted_facts_test = []
    for onesample in tousetestdata:
        q_text = onesample["question"].split("\n\n")[-1].replace("\nResponse: ", "")
        cur_q = instruction + "\n" + q_text
        if onesample["label"] == 1:
            all_contradicted_facts_test.append(cur_q)
        elif onesample["label"] == 0:
            all_supported_facts_test.append(cur_q)
        else:
            print("test label error!")
    print(len(all_supported_facts_train), len(all_contradicted_facts_train), len(all_supported_facts_test), len(all_contradicted_facts_test))
    
    model_id = "/data/home/LLaMA-Factory/models/llama3_8b_automized_claim_classifer_lora_sft_target_all_true_false_cot_diff_prompt_qa_format"

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

    all_labels = []
    all_scores = []
    all_score_pos = []
    all_support_result = []
    all_contradicted_result = []
    all_support_query = []
    all_contradicted_query = []
    allaccforq1_support_token = 0
    allaccforq1_contradicted_token = 0
    # for onesupported_question_text_1 in all_supported_facts_train:
    for onesupported_question_text_1 in all_supported_facts_test:
        onedata = onesupported_question_text_1.strip().split("\n")
        assert len(onedata) == 2
        query = onedata[1]
        onlyquery = query.replace("Claim: ", "")
        all_support_query.append(onlyquery)
        support_res1, support_response_q1 = getTrueorFalse(model, tokenizer, onesupported_question_text_1, terminators)
        oneres = support_res1[0] / (support_res1[0] + support_res1[1])      ####seems to be equal to the difference between support_res1[0] and support_res1[1]
        all_support_result.append(oneres)
        all_labels.append(0)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if support_response_q1.strip().lower() == 'true':
            allaccforq1_support_token += 1
        
    # for onecontradicted_question_text_1 in all_contradicted_facts_train:
    for onecontradicted_question_text_1 in all_contradicted_facts_test:
        onedata = onecontradicted_question_text_1.strip().split("\n")
        assert len(onedata) == 2
        query = onedata[1]
        onlyquery = query.replace("Claim: ", "")
        all_contradicted_query.append(onlyquery)
        contradicted_res1, contradicted_response_q1 = getTrueorFalse(model, tokenizer, onecontradicted_question_text_1, terminators)
        oneres = contradicted_res1[0] / (contradicted_res1[0] + contradicted_res1[1])
        all_contradicted_result.append(oneres)
        all_labels.append(1)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        if contradicted_response_q1.strip().lower() == 'false':
            allaccforq1_contradicted_token += 1    
    
    print(len(all_supported_facts_test), allaccforq1_support_token, len(all_contradicted_facts_test), allaccforq1_contradicted_token)
    precision_direct = allaccforq1_contradicted_token / (allaccforq1_contradicted_token + len(all_supported_facts_test) - allaccforq1_support_token)
    recall_direct = allaccforq1_contradicted_token / len(all_contradicted_facts_test)
    f1_direct = 2 * precision_direct * recall_direct / (precision_direct + recall_direct)
    print("precison: ", precision_direct)
    print("recall: ", recall_direct)
    print("f1_direct: ", f1_direct)

    correct_acc_direct = allaccforq1_support_token / len(all_supported_facts_test)
    wrong_acc_direct = allaccforq1_contradicted_token / len(all_contradicted_facts_test)
    balance_acc_direct = (correct_acc_direct + wrong_acc_direct) / 2
    print("correct accuracy direct: ", correct_acc_direct)
    print("wrong accurayc direct:", wrong_acc_direct)
    print("balanced accuracy direct: ", balance_acc_direct)
    

    all_label_tensor = torch.tensor(all_labels)
    all_score_tensor = torch.tensor(all_scores)
    auroc = compute_auroc(all_label_tensor, all_score_tensor)
    auc_sklearn = roc_auc_score(all_labels, all_score_pos)
    print("auroc: ", auroc, " auc_sklearn: ", auc_sklearn)

    ### plot
    ### ******************************************************
    plot_res(all_support_result, all_contradicted_result)
    ### ******************************************************

    # all_thresholds = [0.05 * i for i in range(10, 20)]
    all_thresholds = [0.02 * i for i in range(15, 36)]
    # all_thresholds.extend([0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999])
    print(all_thresholds)
    
    best_acc = -1
    best_threshold = -1.0
    threshold_for_case_study = 0.54
    all_support_answer = []
    all_contradicted_answer = []
    for threshold in all_thresholds:
        print("*********************************************************************************************************************")
        print("threshold: ", threshold)
        supportfact_predict_as_true = 0
        supportfact_predict_as_false = 0
        contradictedfact_predict_as_true = 0
        contradictedfact_predict_as_false = 0
        for oneres in all_support_result:
            if oneres >= threshold:
                supportfact_predict_as_true += 1
                if threshold == threshold_for_case_study:
                    all_support_answer.append("true")
            else:
                supportfact_predict_as_false += 1
                if threshold == threshold_for_case_study:
                    all_support_answer.append("false")
        for oneres in all_contradicted_result:
            if oneres < threshold:
                contradictedfact_predict_as_false += 1
                if threshold == threshold_for_case_study:
                    all_contradicted_answer.append("false")
            else:
                contradictedfact_predict_as_true += 1
                if threshold == threshold_for_case_study:
                    all_contradicted_answer.append("true")

        # print("P(True)/(P(True)+P(False)) support accuracy: ", supportfact_predict_as_true, len(all_support_result), supportfact_predict_as_true / len(all_support_result))
        # print("P(True)/(P(True)+P(False)) contradicted accuracy: ", contradictedfact_predict_as_false, len(all_contradicted_result), contradictedfact_predict_as_false / len(all_contradicted_result))
        
        # print(supportfact_predict_as_true, supportfact_predict_as_false, contradictedfact_predict_as_false, contradictedfact_predict_as_true)
        precision = contradictedfact_predict_as_false / (contradictedfact_predict_as_false + supportfact_predict_as_false) if (contradictedfact_predict_as_false + supportfact_predict_as_false) > 0 else 0
        recall = contradictedfact_predict_as_false / len(all_contradicted_result)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        print("precision: ", format(precision, '.3f'))
        print("recall: ", format(recall, '.3f'))
        print("f1_score: ", format(f1_score, '.3f'))

        correct_acc = supportfact_predict_as_true / len(all_support_result)
        wrong_acc = contradictedfact_predict_as_false / len(all_contradicted_result)
        balance_acc = (correct_acc + wrong_acc) / 2
        print("correct accuracy: ", format(correct_acc, '.3f'))
        print("wrong accurayc:", format(wrong_acc, '.3f'))
        print("balanced accuracy: ", format(balance_acc, '.3f'))
        if balance_acc > best_acc:
            best_acc = balance_acc
            best_threshold = threshold
        print("*********************************************************************************************************************")
    print("best_acc: ", best_acc)
    print("best_threshold: ", best_threshold)
    print(len(all_support_query), len(all_support_answer))
    print(len(all_contradicted_query), len(all_contradicted_answer))

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

# handle_decomposed_facts()

# handle_decomposed_facts_cot_true_false()
# handle_decomposed_facts_cot_true_false_new()

handle_decomposed_facts_true_false_cot()