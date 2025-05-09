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
from tqdm import tqdm


class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.eval()
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device-auto")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                # print(prompt)
                input_tokens_chat = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False
                )
                # print(input_tokens_chat)
                
                inputs = self.tokenizer(input_tokens_chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
                generate_ids = self.model.generate(
                    inputs,
                    max_new_tokens=5,
                    eos_token_id=self.terminators,
                    do_sample=False
                )
                # print(generate_ids)
                newoutput = generate_ids[0][inputs.shape[-1]:]
                # print(newoutput)
                output_text = self.tokenizer.decode(
                    newoutput, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                # print(output_text)
                score_ = self.text_postprocessing(output_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]


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
        0: "SelfCheckGPT",
    }
    alltoplot = [[all_support_result, all_contradicted_result]]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
    tousesupportedres = alltoplot[0][0]
    tousecontradictedres = alltoplot[0][1]
    for i in range(len(tousesupportedres)):
        if tousesupportedres[i] == 1.0:
            tousesupportedres[i] -= 0.001
    for i in range(len(tousecontradictedres)):
        if tousecontradictedres[i] == 1.0:
            tousecontradictedres[i] -= 0.001
    tousesupportedres.sort()
    tousecontradictedres.sort()

    bins = np.linspace(0, 1, 21)
    
    data_bucketed_s = np.digitize(tousesupportedres, bins)
    allfigres_s = [0 for _ in range(len(bins))]
    for onedata in data_bucketed_s:
        allfigres_s[onedata] += 1
    max_allfigres_s = max(allfigres_s)
    axs.bar(bins, allfigres_s, color='blue', alpha=0.5, label='Correct Claim', width=0.05)

    data_bucketed_c = np.digitize(tousecontradictedres, bins)
    allfigres_c = [0 for _ in range(len(bins))]
    for onedata in data_bucketed_c:
        allfigres_c[onedata] += 1
    max_allfigres_c = max(allfigres_c)
    max_y = max(max_allfigres_s, max_allfigres_c)
    # print(max_y)
    axs.bar(bins, allfigres_c, color='orange', alpha=0.5, label='Wrong Claim', width=0.05)

    axs.set_xticks(np.arange(0, 1, 0.2))
    axs.set_yticks(np.arange(0, max_y + 5, 20))

    axs.set_xlabel(idx2name[0], fontsize=20)
    axs.set_ylabel("Counts", fontsize=20)

    axs.legend(fontsize=16)
    pictitle = "SelfCheckGPT"
    plt.suptitle(pictitle, fontsize=32)
    picname = "Figures/SelfCheckGPT" + ".png"
    plt.savefig(picname)

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
    # tousetestdata = test_data
    tousetestdata = valid_data
    print(len(tousetraindata), len(tousetestdata))
    all_supported_facts_train = []
    all_contradicted_facts_train = []
    for onesample in tousetraindata:
        if onesample["label"] == 1:
            all_contradicted_facts_train.append(onesample["question"])
        elif onesample["label"] == 0:
            all_supported_facts_train.append(onesample["question"])
        else:
            print("train label error!")

    all_supported_facts_test = []
    all_contradicted_facts_test = []
    for onesample in tousetestdata:
        if onesample["label"] == 1:
            all_contradicted_facts_test.append(onesample["question"])
        elif onesample["label"] == 0:
            all_supported_facts_test.append(onesample["question"])
        else:
            print("test label error!")
    print(len(all_supported_facts_train), len(all_contradicted_facts_train), len(all_supported_facts_test), len(all_contradicted_facts_test))

    claim2qfile = "./SelfCheckGPT/selfcheckdata.json"
    allclaim2q = {}
    with open(claim2qfile,'r') as f:
        allclaim2q = json.load(f)
    print(len(allclaim2q))

    q2samplesfile = "./SelfCheckGPT/longfact_selfcheckgpt_stochastic_samples.json"
    allq2samples = {}
    with open(q2samplesfile,'r') as f:
        alllines = f.readlines()
        for oneline in alllines:
            json_object = json.loads(oneline.strip())
            prompt, response = json_object["prompt"], json_object["response"]
            prompt = prompt.split("user<|end_header_id|>")[1].split("<|eot_id|><|start_header_id|>assistant")[0].strip()
            allq2samples[prompt] = response
    print(len(allq2samples))
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    selfcheck_prompt = SelfCheckLLMPrompt(model_id)

    all_labels = []
    all_scores = []
    all_score_pos = []
    all_support_result = []
    all_contradicted_result = []
    idx = 0
    # for onesupported_question_text_1 in all_supported_facts_train:
    for onesupported_question_text_1 in all_supported_facts_test:
        oneclaim = onesupported_question_text_1.split("\n\nClaim:")[-1].split("\nResponse:")[0].strip()
        # print(oneclaim)
        assert oneclaim in allclaim2q.keys()
        onequery = allclaim2q[oneclaim]
        # print(onequery)
        # print(len(allq2samples.keys()))
        assert onequery in allq2samples.keys()
        allsamples = allq2samples[onequery]
        sent_scores_prompt = selfcheck_prompt.predict(
            sentences = [oneclaim],                          # list of sentences
            sampled_passages = allsamples, # list of sampled passages
            verbose = True, # whether to show a progress bar
        )
        # print(sent_scores_prompt)
        oneres = 1.0 - sent_scores_prompt[0]
        all_support_result.append(oneres)
        all_labels.append(0)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        print(idx)
        idx += 1
    
    idx = 0
    # for onecontradicted_question_text_1 in all_contradicted_facts_train:
    for onecontradicted_question_text_1 in all_contradicted_facts_test:
        oneclaim = onecontradicted_question_text_1.split("\n\nClaim:")[-1].split("\nResponse:")[0].strip()
        # print(oneclaim)
        assert oneclaim in allclaim2q.keys()
        onequery = allclaim2q[oneclaim]
        # print(onequery)
        # print(len(allq2samples.keys()))
        assert onequery in allq2samples.keys()
        allsamples = allq2samples[onequery]
        sent_scores_prompt = selfcheck_prompt.predict(
            sentences = [oneclaim],                          # list of sentences
            sampled_passages = allsamples, # list of sampled passages
            verbose = True, # whether to show a progress bar
        )
        # print(sent_scores_prompt)
        oneres = 1.0 - sent_scores_prompt[0]
        all_contradicted_result.append(oneres)
        all_labels.append(1)
        all_scores.append([oneres, 1.0 - oneres])
        all_score_pos.append(1.0 - oneres)
        print(idx)
        idx += 1

    all_label_tensor = torch.tensor(all_labels)
    all_score_tensor = torch.tensor(all_scores)
    auroc = compute_auroc(all_label_tensor, all_score_tensor)
    auc_sklearn = roc_auc_score(all_labels, all_score_pos)
    print("auroc: ", auroc, " auc_sklearn: ", auc_sklearn)

    onefile = "./SelfCheckGPT/validres.json"
    touseres = {
        "all_support_result": all_support_result,
        "all_contradicted_result": all_contradicted_result
    }
    # with open(onefile, "w+") as f:
    #     json.dump(touseres, f, indent=4, separators=(",", ": "))

    ### plot
    ### ******************************************************
    # plot_res(all_support_result, all_contradicted_result)
    ### ******************************************************

    all_thresholds = [0.05 * i for i in range(10, 20)]
    print(all_thresholds)
    
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
        print("*********************************************************************************************************************")

def get_unknown_result():
    resfile = "./SelfCheckGPT/validres.json"
    allres = {}
    with open(resfile,'r') as f:
        allres = json.load(f)
    all_support_result = allres["all_support_result"]
    all_contradicted_result = allres["all_contradicted_result"]

    lowthresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    highthresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    best_acc = -1
    best_low = -1.0
    best_high = -1.0
    basic_acc = 0.75
    # for threshold in all_thresholds:
    for low_threshold in lowthresholds:
        for high_threshold in highthresholds:
            print("*********************************************************************************************************************")
            print("threshold: ", [low_threshold, high_threshold])
            supportfact_predict_as_true = 0
            supportfact_predict_as_false = 0
            supportfact_predict_as_unknown = 0
            contradictedfact_predict_as_true = 0
            contradictedfact_predict_as_false = 0
            contradictedfact_predict_as_unknown = 0
            
            for oneres in all_support_result:
                if oneres >= high_threshold:
                    supportfact_predict_as_true += 1
                elif oneres <= low_threshold:
                    supportfact_predict_as_false += 1
                else:
                    supportfact_predict_as_unknown += 1
            for oneres in all_contradicted_result:
                if oneres >= high_threshold:
                    contradictedfact_predict_as_true += 1
                elif oneres <= low_threshold:
                    contradictedfact_predict_as_false += 1
                else:
                    contradictedfact_predict_as_unknown += 1

            print(supportfact_predict_as_true, supportfact_predict_as_false, supportfact_predict_as_unknown, contradictedfact_predict_as_true, contradictedfact_predict_as_false, contradictedfact_predict_as_unknown)
            support_accuracy = supportfact_predict_as_true / len(all_support_result)
            support_add_unknown_accuracy = (supportfact_predict_as_true + supportfact_predict_as_unknown) / len(all_support_result)
            contradicted_accuracy = contradictedfact_predict_as_false / len(all_contradicted_result)
            contradicted_add_unknown_accuracy = (contradictedfact_predict_as_false + contradictedfact_predict_as_unknown) / len(all_contradicted_result)
            print("support accuracy: ", support_accuracy)
            print("support add unknown accuracy: ", support_add_unknown_accuracy)
            print("contradicted accuracy: ", contradicted_accuracy)
            print("contradicted add unknown accuracy: ", contradicted_add_unknown_accuracy)
            print("average accuray: ", (support_accuracy + contradicted_accuracy) / 2)
            print("average add unknown accuracy: ", (support_add_unknown_accuracy + contradicted_add_unknown_accuracy) / 2)
            print("final average: ", (support_accuracy + contradicted_accuracy + support_add_unknown_accuracy + contradicted_add_unknown_accuracy) / 4)
            print("*********************************************************************************************************************")

            if support_add_unknown_accuracy >= basic_acc and contradicted_add_unknown_accuracy >= basic_acc:
                tmpacc = (support_accuracy + contradicted_accuracy) / 2
                if tmpacc > best_acc:
                    best_acc = tmpacc
                    best_low = low_threshold
                    best_high = high_threshold
    print("best_acc: ", best_acc)
    print("best_low: ", best_low)
    print("best_high: ", best_high)

def get_balanced_acc():
    resfile = "./SelfCheckGPT/validres.json"
    allres = {}
    with open(resfile,'r') as f:
        allres = json.load(f)
    all_support_result = allres["all_support_result"]
    all_contradicted_result = allres["all_contradicted_result"]

    # all_thresholds = [0.05 * i for i in range(10, 20)]
    all_thresholds = [0.02 * i for i in range(15, 36)]
    # all_thresholds.extend([0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999])
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

# handle_decomposed_facts()
# get_unknown_result()
get_balanced_acc()