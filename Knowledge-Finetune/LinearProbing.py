import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from dataset import TrainDataset
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import random
import matplotlib.pyplot as plt
from openai import OpenAI
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def calculate_f1_score(predictions, labels):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Calculate true positives, false positives, and false negatives
    for pred, label in zip(predictions, labels):
        if pred == 1 and label == 1:
            true_positive += 1
        elif pred == 1 and label == 0:
            false_positive += 1
        elif pred == 0 and label == 1:
            false_negative += 1

    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, precision, recall

def binary_eval(predy, testy):
    acc = accuracy_score(testy, predy)
    f1 = f1_score(testy, predy)
    myf1, myprecision, myrecall = calculate_f1_score(predy, testy)
    # print(f1, myf1)
    return acc, myf1, myprecision, myrecall
    # return acc, myprecision, myrecall

def calibration(y, class_pred, conf, num_bins=10):
    """Compute the calibration.

    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263

    Args:
      y: one-hot encoding of the true classes, size (?, num_classes)
      p_mean: numpy array, size (?, num_classes)
             containing the mean output predicted probabilities
      num_bins: number of bins

    Returns:
      ece: Expected Calibration Error
      mce: Maximum Calibration Error
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        class_pred = class_pred.cpu().numpy()
        conf = conf.cpu().numpy()
    # Compute for every test sample x, the predicted class.
    # class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    # conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    # y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins + 1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    if len(nb_items_bin) == 0:
        logging.warning("ECE computation failed.")
        return float("nan"), float("nan")

    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(float) / np.sum(nb_items_bin),
    )
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))
    return ece, mce

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

def compute_uncertainty_metrics(labels, logits):
    """
    Arguments:
        labels: Shape (N,)
        logits: Shape (N, 2)
    """
    p = logits.softmax(dim=-1)

    pred = p.argmax(dim=-1)
    acc = (pred == labels).float().mean(dim=0)

    ece, _ = calibration(
        labels,
        pred,
        p[torch.arange(p.size(0)), pred].float(),
    )

    auroc = compute_auroc(labels, p)

    return auroc, ece

def get_data(data_path, hidden_state_type):
    
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
        if hidden_state_type == 1:
            touseright = right_final_layer_final_token[i]
            tousehallu = hallu_final_layer_final_token[i]
        elif hidden_state_type == 2:
            touseright = right_all_layers_final_token[i]
            tousehallu = hallu_all_layers_final_token[i]
        elif hidden_state_type == 3:
            touseright = right_final_layer_all_tokens[i]
            tousehallu = hallu_final_layer_all_tokens[i]
        elif hidden_state_type == 4:
            touseright = right_final_layer_final_and_all_tokens[i]
            tousehallu = hallu_final_layer_final_and_all_tokens[i]
        else:
            print("hidden state type not supported!")
            exit -1
        enddata.append({
            "fact": all_support_text[i]["fact"],
            "question": all_support_text[i]["question"],
            "hidden": touseright,
            "label": 0
        })
        enddata.append({
            "fact": all_contradicted_text[i]["fact"],
            "question": all_contradicted_text[i]["question"],
            "hidden": tousehallu,
            "label": 1
        })
        
    return enddata

class Model():
    def __init__(self, args, path=None):
        self.args = args
        input_size = args.input_size
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(args.dropout))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 128))
        self.model.add_module(f"relu2", nn.ReLU())
        self.model.add_module(f"linear3", nn.Linear(128, 64))
        self.model.add_module(f"relu3", nn.ReLU())
        self.model.add_module(f"linear4", nn.Linear(64, 2))
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location = "cpu")["model_state_dict"])
        self.model.to(args.device)
        
    
    def save(self, auc, ei, prefix, name):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "valid_auc": auc,
                    "epoch": ei},
                    prefix + f"{name}_model.pt")
        

    def run(self, optim):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
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

        train_dataset = TrainDataset(train_data, self.args)
        valid_dataset = TrainDataset(valid_data, self.args, typ="valid")
        test_dataset = TrainDataset(test_data, self.args, typ="valid")
        train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                               batch_size=self.args.batch_size // 2,
                                               shuffle=False,
                                               num_workers=2)
        test_dataloader = DataLoader(dataset=test_dataset,
                                               batch_size=self.args.batch_size // 2,
                                               shuffle=False,
                                               num_workers=2)
        
        nSamples = [len(train_dataset) - train_dataset.halu_num, train_dataset.halu_num]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
        loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
        # loss_func = nn.CrossEntropyLoss().to(self.args.device)
        
        best_f1 = -1
        best_precision = -1
        best_recall = -1
        best_acc = -1
        best_auc = -1
        best_ece = -1
        best_epoch = [0]
        for ei in range(epoch_start, epoch+1):
            cnt = 0
            self.model.train()
            train_loss = 0
            predy, trainy, hallu_sm_score = [], [], []
            for step, batch in enumerate(train_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                if step == 0:
                    all_label_ids_train = label_ids.detach().cpu()
                else:
                    all_label_ids_train = torch.cat((all_label_ids_train, label_ids.detach().cpu()), dim=0)
                # print(all_label_ids_train.shape)
                score = self.model(input_)
                if step == 0:
                    all_score_train = score.detach().cpu()
                else:
                    all_score_train = torch.cat((all_score_train, score.detach().cpu()), dim=0)
                # print(all_score_train.shape)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)

                trainy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                train_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                cnt += 1
                if cnt % 10 == 0:
                    print("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))
            print("Training Epoch {} ...".format(ei))
            train_acc, train_f1, train_precision, train_recall = binary_eval(predy, trainy)
            # train_auc = roc_auc_score(trainy, hallu_sm_score)
            train_auc, train_ece = compute_uncertainty_metrics(all_label_ids_train, all_score_train)
            
            print("Train Epoch {} end ! Loss : {}; Train f1: {}; Train precision: {}; Train recall: {}; Train accuracy: {}; Train auc: {}".format(ei, train_loss, train_f1, train_precision, train_recall, train_acc, train_auc))

            self.model.eval()
            predy, validy, hallu_sm_score = [], [], []
            valid_loss = 0
            with torch.no_grad():
                for step, batch in enumerate(valid_dataloader):
                    input_ = batch["input"].to(self.args.device)
                    label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                    if step == 0:
                        all_label_ids_valid = label_ids.detach().cpu()
                    else:
                        all_label_ids_valid = torch.cat((all_label_ids_valid, label_ids.detach().cpu()), dim=0)
                    score = self.model(input_)
                    if step == 0:
                        all_score_valid = score.detach().cpu()
                    else:
                        all_score_valid = torch.cat((all_score_valid, score.detach().cpu()), dim=0)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    validy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    valid_loss += loss.item()
            print("Valid Epoch {} ...".format(ei))

            valid_acc, valid_f1, valid_precision, valid_recall = binary_eval(predy, validy)
            # valid_auc = roc_auc_score(validy, hallu_sm_score)
            valid_auc, valid_ece = compute_uncertainty_metrics(all_label_ids_valid, all_score_valid)

            if valid_auc > best_auc:
                best_auc = valid_auc
                best_f1 = valid_f1
                best_precision = valid_precision
                best_recall = valid_recall
                best_acc = valid_acc
                best_epoch[0] = ei
                self.save(valid_auc, ei, prefix, "best_auc")
                
        self.save(valid_auc, ei, prefix, "last")
        print(f"Best auc : {best_auc} from epoch {best_epoch[0]}th")

        print(f"Start test for best epoch {best_epoch[0]}th")
        bestpath = prefix + "best_auc_model.pt"
        self.model.load_state_dict(torch.load(bestpath, map_location = "cpu")["model_state_dict"])
        self.model.eval()
        predy, testy, hallu_sm_score = [], [], []
        test_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                if step == 0:
                    all_label_ids_test = label_ids.detach().cpu()
                else:
                    all_label_ids_test = torch.cat((all_label_ids_test, label_ids.detach().cpu()), dim=0)
                score = self.model(input_)
                if step == 0:
                    all_score_test = score.detach().cpu()
                else:
                    all_score_test = torch.cat((all_score_test, score.detach().cpu()), dim=0)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)
                testy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                test_loss += loss.item()
        test_acc, test_f1, test_precision, test_recall = binary_eval(predy, testy)
        # test_auc = roc_auc_score(testy, hallu_sm_score)
        test_auc, test_ece = compute_uncertainty_metrics(all_label_ids_test, all_score_test)
        print("*********************************************************************************************************************")
        print("Seed: ", self.args.seed)
        print("Train Epoch: ", self.args.train_epoch)
        print("Batch Size: ", self.args.batch_size)
        print("Learning Rate: ", self.args.lr)
        print("Hidden State Type: ", self.args.hidden_state_type)
        print(f"Best validation auc : {best_auc} for the best epoch {best_epoch[0]}th")
        print(f"Test auc : {test_auc}, Test acc : {test_acc}, Test f1 : {test_f1}, Test precision : {test_precision}, Test recall : {test_recall} for the best epoch {best_epoch[0]}th")
        print("*********************************************************************************************************************")


    def run_threshold_search(self, optim):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
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

        train_dataset = TrainDataset(train_data, self.args)
        valid_dataset = TrainDataset(valid_data, self.args, typ="valid")
        test_dataset = TrainDataset(test_data, self.args, typ="valid")
        train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                               batch_size=self.args.batch_size // 2,
                                               shuffle=False,
                                               num_workers=2)
        test_dataloader = DataLoader(dataset=test_dataset,
                                               batch_size=self.args.batch_size // 2,
                                               shuffle=False,
                                               num_workers=2)
        
        nSamples = [len(train_dataset) - train_dataset.halu_num, train_dataset.halu_num]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
        loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
        # loss_func = nn.CrossEntropyLoss().to(self.args.device)
        
        best_f1 = -1
        best_precision = -1
        best_recall = -1
        best_acc = -1
        best_auc = -1
        best_ece = -1
        best_epoch = [0]
        for ei in range(epoch_start, epoch+1):
            cnt = 0
            self.model.train()
            train_loss = 0
            predy, trainy, hallu_sm_score = [], [], []
            for step, batch in enumerate(train_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                if step == 0:
                    all_label_ids_train = label_ids.detach().cpu()
                else:
                    all_label_ids_train = torch.cat((all_label_ids_train, label_ids.detach().cpu()), dim=0)
                # print(all_label_ids_train.shape)
                score = self.model(input_)
                if step == 0:
                    all_score_train = score.detach().cpu()
                else:
                    all_score_train = torch.cat((all_score_train, score.detach().cpu()), dim=0)
                # print(all_score_train.shape)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                _, pred = torch.max(score, dim=1)

                trainy.extend(label_ids.tolist())
                predy.extend(pred.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                train_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                cnt += 1
                if cnt % 10 == 0:
                    print("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))
            print("Training Epoch {} ...".format(ei))
            train_acc, train_f1, train_precision, train_recall = binary_eval(predy, trainy)
            # train_auc = roc_auc_score(trainy, hallu_sm_score)
            train_auc, train_ece = compute_uncertainty_metrics(all_label_ids_train, all_score_train)
            
            print("Train Epoch {} end ! Loss : {}; Train f1: {}; Train precision: {}; Train recall: {}; Train accuracy: {}; Train auc: {}".format(ei, train_loss, train_f1, train_precision, train_recall, train_acc, train_auc))

            self.model.eval()
            predy, validy, hallu_sm_score = [], [], []
            valid_loss = 0
            with torch.no_grad():
                for step, batch in enumerate(valid_dataloader):
                    input_ = batch["input"].to(self.args.device)
                    label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                    if step == 0:
                        all_label_ids_valid = label_ids.detach().cpu()
                    else:
                        all_label_ids_valid = torch.cat((all_label_ids_valid, label_ids.detach().cpu()), dim=0)
                    score = self.model(input_)
                    if step == 0:
                        all_score_valid = score.detach().cpu()
                    else:
                        all_score_valid = torch.cat((all_score_valid, score.detach().cpu()), dim=0)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    validy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    valid_loss += loss.item()
            print("Valid Epoch {} ...".format(ei))

            valid_acc, valid_f1, valid_precision, valid_recall = binary_eval(predy, validy)
            # valid_auc = roc_auc_score(validy, hallu_sm_score)
            valid_auc, valid_ece = compute_uncertainty_metrics(all_label_ids_valid, all_score_valid)

            if valid_auc > best_auc:
                best_auc = valid_auc
                best_f1 = valid_f1
                best_precision = valid_precision
                best_recall = valid_recall
                best_acc = valid_acc
                best_epoch[0] = ei
                self.save(valid_auc, ei, prefix, "best_auc")
                
        self.save(valid_auc, ei, prefix, "last")
        print(f"Best auc : {best_auc} from epoch {best_epoch[0]}th")

        print(f"Start test for best epoch {best_epoch[0]}th")
        bestpath = prefix + "best_auc_model.pt"
        self.model.load_state_dict(torch.load(bestpath, map_location = "cpu")["model_state_dict"])
        self.model.eval()
        testy, predy, hallu_sm_score = [], [], []
        all_thresholds = [0.05 * i for i in range(0, 20)]
        # all_thresholds.extend([0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999])
        all_predy = [[] for _ in range(len(all_thresholds))]
        test_loss = 0
        all_support_result = []
        all_contradicted_result = []
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
            # for step, batch in enumerate(valid_dataloader):
                input_ = batch["input"].to(self.args.device)
                label_ids = torch.LongTensor([k[0] for k in batch["y"].tolist()]).to(self.args.device)
                if step == 0:
                    all_label_ids_test = label_ids.detach().cpu()
                else:
                    all_label_ids_test = torch.cat((all_label_ids_test, label_ids.detach().cpu()), dim=0)
                score = self.model(input_)
                if step == 0:
                    all_score_test = score.detach().cpu()
                else:
                    all_score_test = torch.cat((all_score_test, score.detach().cpu()), dim=0)
                hallu_sm = F.softmax(score, dim=1)[:, 1]
                testy.extend(label_ids.tolist())
                hallu_sm_score.extend(hallu_sm.tolist())
                loss = loss_func(score, label_ids)
                test_loss += loss.item()

                score_sm_all = F.softmax(score, dim=1)
                for gg in range(len(all_thresholds)):
                    one_predy = []
                    for kk in range(score_sm_all.shape[0]):
                        if score_sm_all[kk][1] >= all_thresholds[gg]:
                            one_predy.append(1)
                        else:
                            one_predy.append(0)
                    all_predy[gg].extend(one_predy)

                score_sm_all_cpu = score_sm_all.detach().cpu()
                label_ids_list = label_ids.tolist()
                for kk in range(score_sm_all_cpu.shape[0]):
                    # print(label_ids_list[kk] == 1)
                    if label_ids_list[kk] == 1:
                        all_contradicted_result.append(score_sm_all_cpu[kk][0] / (score_sm_all_cpu[kk][0] + score_sm_all_cpu[kk][1]))
                    else:
                        all_support_result.append(score_sm_all_cpu[kk][0] / (score_sm_all_cpu[kk][0] + score_sm_all_cpu[kk][1]))

                _, pred = torch.max(score, dim=1)
                predy.extend(pred.tolist())
        
        #### plot
        #### ******************************************************
        # print(len(all_support_result), len(all_contradicted_result))
        # self.plot_res(all_support_result, all_contradicted_result)
        #### ******************************************************

        print(len(all_predy), len(all_predy[0]))
        test_acc, test_f1, test_precision, test_recall = binary_eval(predy, testy)
        correct_num, correct_acc_num, wrong_num, wrong_acc_num = 0, 0, 0, 0
        for i in range(len(testy)):
            if testy[i] == 0:
                correct_num += 1
                if predy[i] == testy[i]:
                    correct_acc_num += 1
            else:
                wrong_num += 1
                if predy[i] == testy[i]:
                    wrong_acc_num += 1
        print("Direct results!")
        print(correct_acc_num, correct_num, wrong_acc_num, wrong_num)
        correct_acc_direct = correct_acc_num / correct_num
        wrong_acc_direct = wrong_acc_num / wrong_num
        balance_acc_direct = (correct_acc_direct + wrong_acc_direct) / 2
        print("correct accuracy direct: ", correct_acc_direct)
        print("wrong accurayc direct:", wrong_acc_direct)
        print("balanced accuracy direct: ", balance_acc_direct)
        # test_auc = roc_auc_score(testy, hallu_sm_score)
        test_auc, test_ece = compute_uncertainty_metrics(all_label_ids_test, all_score_test)
        print("*********************************************************************************************************************")
        print("Seed: ", self.args.seed)
        print("Train Epoch: ", self.args.train_epoch)
        print("Batch Size: ", self.args.batch_size)
        print("Learning Rate: ", self.args.lr)
        print("Hidden State Type: ", self.args.hidden_state_type)
        print(f"Best validation auc : {best_auc} for the best epoch {best_epoch[0]}th")
        print(f"Test auc : {test_auc} for the best epoch {best_epoch[0]}th")
        print(f"Test acc : {test_acc}, Test f1 : {test_f1}, Test precision : {test_precision}, Test recall : {test_recall} for the best epoch {best_epoch[0]}th")
        print("*********************************************************************************************************************")
        print(f"Start threshold search")
        for gg in range(len(all_thresholds)):
            test_acc, test_f1, test_precision, test_recall = binary_eval(all_predy[gg], testy)
            print(f"Threshold : {all_thresholds[gg]}, Test acc : {test_acc}, Test f1 : {test_f1}, Test precision : {test_precision}, Test recall : {test_recall} for the best epoch {best_epoch[0]}th")
        print("*********************************************************************************************************************")

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

        # print("searching best thresholds for unknown")
        # lowthresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        # highthresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        # best_acc = -1
        # best_low = -1.0
        # best_high = -1.0
        # basic_acc = 0.85
        # # for threshold in all_thresholds:
        # for low_threshold in lowthresholds:
        #     for high_threshold in highthresholds:
        #         print("*********************************************************************************************************************")
        #         print("threshold: ", [low_threshold, high_threshold])
        #         supportfact_predict_as_true = 0
        #         supportfact_predict_as_false = 0
        #         supportfact_predict_as_unknown = 0
        #         contradictedfact_predict_as_true = 0
        #         contradictedfact_predict_as_false = 0
        #         contradictedfact_predict_as_unknown = 0
                
        #         for oneres in all_support_result:
        #             if oneres >= high_threshold:
        #                 supportfact_predict_as_true += 1
        #             elif oneres <= low_threshold:
        #                 supportfact_predict_as_false += 1
        #             else:
        #                 supportfact_predict_as_unknown += 1
        #         for oneres in all_contradicted_result:
        #             if oneres >= high_threshold:
        #                 contradictedfact_predict_as_true += 1
        #             elif oneres <= low_threshold:
        #                 contradictedfact_predict_as_false += 1
        #             else:
        #                 contradictedfact_predict_as_unknown += 1

        #         print(supportfact_predict_as_true, supportfact_predict_as_false, supportfact_predict_as_unknown, contradictedfact_predict_as_true, contradictedfact_predict_as_false, contradictedfact_predict_as_unknown)
        #         support_accuracy = supportfact_predict_as_true / len(all_support_result)
        #         support_add_unknown_accuracy = (supportfact_predict_as_true + supportfact_predict_as_unknown) / len(all_support_result)
        #         contradicted_accuracy = contradictedfact_predict_as_false / len(all_contradicted_result)
        #         contradicted_add_unknown_accuracy = (contradictedfact_predict_as_false + contradictedfact_predict_as_unknown) / len(all_contradicted_result)
        #         print("support accuracy: ", support_accuracy)
        #         print("support add unknown accuracy: ", support_add_unknown_accuracy)
        #         print("contradicted accuracy: ", contradicted_accuracy)
        #         print("contradicted add unknown accuracy: ", contradicted_add_unknown_accuracy)
        #         print("average accuray: ", (support_accuracy + contradicted_accuracy) / 2)
        #         print("average add unknown accuracy: ", (support_add_unknown_accuracy + contradicted_add_unknown_accuracy) / 2)
        #         print("final average: ", (support_accuracy + contradicted_accuracy + support_add_unknown_accuracy + contradicted_add_unknown_accuracy) / 4)
        #         print("*********************************************************************************************************************")

        #         if support_add_unknown_accuracy >= basic_acc and contradicted_add_unknown_accuracy >= basic_acc:
        #             tmpacc = (support_accuracy + contradicted_accuracy) / 2
        #             if tmpacc > best_acc:
        #                 best_acc = tmpacc
        #                 best_low = low_threshold
        #                 best_high = high_threshold
        # print("best_acc: ", best_acc)
        # print("best_low: ", best_low)
        # print("best_high: ", best_high)
    
    def plot_res(self, all_support_result, all_contradicted_result):
        idx2name = {
            0: "P(True)/(P(True)+P(False))",
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
        pictitle = "Linear Probing"
        plt.suptitle(pictitle, fontsize=32)
        picname = "Figures/True_False_linear_probing" + ".png"
        plt.savefig(picname)

    def save_data(self):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
        print(train_index, valid_index)

        train_data = all_data[0:train_index]
        valid_data = all_data[train_index:valid_index]
        test_data = all_data[valid_index:]

        print(len(train_data), len(valid_data), len(test_data))
        print(train_data[0]["question"])
        print(valid_data[0]["question"])
        alltosavetrain = []
        for i in range(len(train_data)):
            onedata = train_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            instruction = onedata[0]
            query = onedata[2]
            if train_data[i]["label"] == 0:
                response = "True"
            elif train_data[i]["label"] == 1:
                response = "False"
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
            instruction = onedata[0]
            query = onedata[2]
            if valid_data[i]["label"] == 0:
                response = "True"
            elif valid_data[i]["label"] == 1:
                response = "False"
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
        trainpath = "/data/home/LLaMA-Factory/data/automized_claim_train.json"
        validpath = "/data/home/LLaMA-Factory/data/automized_claim_valid.json"
        with open(trainpath, "w+") as f:
            json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
        with open(validpath, "w+") as f:
            json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))

    def getTrueorFalse(self, model, tokenizer, question_text_original, terminators):
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

    def save_data_position_bias(self):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
        print(train_index, valid_index)

        train_data = all_data[0:train_index]
        valid_data = all_data[train_index:valid_index]
        test_data = all_data[valid_index:]

        print(len(train_data), len(valid_data), len(test_data))
        print(train_data[0]["question"])
        print(valid_data[0]["question"])
        
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
        alltosavetrain = []
        alltosavetrain_new = []
        diffnum = 0
        for i in range(len(train_data)):
            onedata = train_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            instruction = onedata[0]
            query = onedata[2]
            if train_data[i]["label"] == 0:
                response = "True"
            elif train_data[i]["label"] == 1:
                response = "False"
            else:
                exit -1
            onetosavetrain = {
                "instruction": instruction,
                "input": query,
                "output": response
            }
            alltosavetrain.append(onetosavetrain)
            #####determin whether to handle position bias
            question_original = train_data[i]["question"]
            question_new = question_original.replace("'True' or 'False'", "'False' or 'True'")
            prob_original, res_original = self.getTrueorFalse(model, tokenizer, question_original, terminators)
            prob_new, res_new = self.getTrueorFalse(model, tokenizer, question_new, terminators)
            if res_original.strip().lower() != res_new.strip().lower():
                diffnum += 1
                onetosavetrain_new = {
                    "instruction": instruction.replace("'True' or 'False'", "'False' or 'True'"),
                    "input": query,
                    "output": response
                }
                alltosavetrain_new.append(onetosavetrain_new)
        print("different number: ", diffnum)
        alltosavetrain.extend(alltosavetrain_new)

        alltosavevalid = []
        for i in range(len(valid_data)):
            onedata = valid_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            instruction = onedata[0]
            query = onedata[2]
            if valid_data[i]["label"] == 0:
                response = "True"
            elif valid_data[i]["label"] == 1:
                response = "False"
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
        trainpath = "/data/home/LLaMA-Factory/data/automized_claim_train_position_bias.json"
        validpath = "/data/home/LLaMA-Factory/data/automized_claim_valid_position_bias.json"
        with open(trainpath, "w+") as f:
            json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
        with open(validpath, "w+") as f:
            json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))

    def callgpt4o(self, client, input2gpt):
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

    def save_data_kto(self):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
        print(train_index, valid_index)

        train_data = all_data[0:train_index]
        valid_data = all_data[train_index:valid_index]
        test_data = all_data[valid_index:]

        print(len(train_data), len(valid_data), len(test_data))
        # print(train_data[0]["question"])
        # print(valid_data[0]["question"])
        api_key = ""
        client = OpenAI(api_key=api_key)
        partprompt1 = "Please generate a question that can be answered with the following response:" 
        partprompt2 = "The question should be clear and concise, and should not include any information that is not relevant to the response."
        alltosavetrain = []
        for i in range(len(train_data)):
            print(i)
            onedata = train_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            # instruction = "Generate a claim that you believe is correct."        ##### maybe we can change it to one instruction related to the claim; can use gpt4o to obtain the instruction
            query = onedata[2].replace("Claim: ", "")
            input2gpt = partprompt1 + " " + query + " " + partprompt2
            # print(input2gpt)
            instruction = self.callgpt4o(client, input2gpt)
            # print(instruction)
            if train_data[i]["label"] == 0:
                response = True
            elif train_data[i]["label"] == 1:
                response = False
            else:
                exit -1
            onetosavetrain = {
                "instruction": instruction,
                "output": query,
                "kto_tag": response
            }
            alltosavetrain.append(onetosavetrain)
        
        alltosavevalid = []
        for i in range(len(valid_data)):
            print(i)
            onedata = valid_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            # instruction = "Generate a claim that you believe is correct."
            query = onedata[2].replace("Claim: ", "")
            input2gpt = partprompt1 + " " + query + " " + partprompt2
            # print(input2gpt)
            instruction = self.callgpt4o(client, input2gpt)
            # print(instruction)
            if valid_data[i]["label"] == 0:
                response = True
            elif valid_data[i]["label"] == 1:
                response = False
            else:
                exit -1
            onetosavevalid = {
                "instruction": instruction,
                "output": query,
                "kto_tag": response
            }
            alltosavevalid.append(onetosavevalid)
        print(len(alltosavetrain), len(alltosavevalid))
        print(alltosavetrain[0])
        print(alltosavevalid[0])
        trainpath = "/data/home/LLaMA-Factory/data/automized_claim_train_kto_new_instruction.json"
        validpath = "/data/home/LLaMA-Factory/data/automized_claim_valid_kto_new_instruction.json"
        with open(trainpath, "w+") as f:
            json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
        with open(validpath, "w+") as f:
            json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))

    def save_data_dpo(self):
        now = datetime.datetime.now()
        prefix = f"{self.args.output_path}/{self.args.model_name}/train_log/"
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        epoch, epoch_start = self.args.train_epoch, 1
        
        all_data = get_data(self.args.data_path, self.args.hidden_state_type)
        random.shuffle(all_data)
        print(len(all_data))
        train_index = int(len(all_data) * self.args.train_split)
        valid_index = int(len(all_data) * (self.args.train_split + self.args.valid_split))
        print(train_index, valid_index)

        train_data = all_data[0:train_index]
        valid_data = all_data[train_index:valid_index]
        test_data = all_data[valid_index:]

        print(len(train_data), len(valid_data), len(test_data))
        print(train_data[0]["question"])
        print(valid_data[0]["question"])
        alltosavetrain = []
        for i in range(len(train_data)):
            onedata = train_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            instruction = onedata[0]
            query = onedata[2]
            if train_data[i]["label"] == 0:
                response = "True"
                wrong_response = "False"
            elif train_data[i]["label"] == 1:
                response = "False"
                wrong_response = "True"
            else:
                exit -1
            onetosavetrain = {
                "instruction": instruction,
                "input": query,
                "chosen": response,
                "rejected": wrong_response
            }
            alltosavetrain.append(onetosavetrain)
        
        alltosavevalid = []
        for i in range(len(valid_data)):
            onedata = valid_data[i]["question"].strip().split("\n")
            assert len(onedata) == 4
            instruction = onedata[0]
            query = onedata[2]
            if valid_data[i]["label"] == 0:
                response = "True"
                wrong_response = "False"
            elif valid_data[i]["label"] == 1:
                response = "False"
                wrong_response = "True"
            else:
                exit -1
            onetosavevalid = {
                "instruction": instruction,
                "input": query,
                "chosen": response,
                "rejected": wrong_response
            }
            alltosavevalid.append(onetosavevalid)
        print(len(alltosavetrain), len(alltosavevalid))
        print(alltosavetrain[0])
        print(alltosavevalid[0])
        trainpath = "/data/home/LLaMA-Factory/data/automized_claim_train_dpo.json"
        validpath = "/data/home/LLaMA-Factory/data/automized_claim_valid_dpo.json"
        with open(trainpath, "w+") as f:
            json.dump(alltosavetrain, f, indent=4, separators=(",", ": "))
        with open(validpath, "w+") as f:
            json.dump(alltosavevalid, f, indent=4, separators=(",", ": "))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct")
    parser.add_argument("--output_path", default="./LinearProbingResult/output", type=str)
    parser.add_argument("--data_path", default="./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy.json", type=str)
    # parser.add_argument("--data_path", default="./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy_noinstruction_onlyclaim.json", type=str)
    # parser.add_argument("--data_path", default="./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy_noinstruction_onlyclaim_with_chattemplate.json", type=str)
    # parser.add_argument("--data_path", default="./HiddenStates/longfact_alldata_hiddenstates-llama3-8b-instruct_greedy_question_and_claim.json", type=str)

    parser.add_argument("--train_epoch", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--input_size", default=4096, type=int)
    parser.add_argument("--train_split", default=0.7, type=float)
    parser.add_argument("--valid_split", default=0.1, type=float)
    parser.add_argument("--hidden_state_type", default=2, type=int)
    
    args = parser.parse_args()
    print("------------------------------------------args------------------------------------------")
    print(args)
    set_seed(args.seed)
    
    model = Model(args)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optim_func = torch.optim.Adam
    named_params = list(model.model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd, 'lr': args.lr},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim_func(optimizer_grouped_parameters)

    model.run(optimizer)
    # model.run_threshold_search(optimizer)
    # model.save_data()
    # model.save_data_kto()
    # model.save_data_position_bias()
    # model.save_data_dpo()
    print(args.model_name)