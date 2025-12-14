import scipy.sparse as sp
import pickle
import sys
from pack1 import pack, pack1
import timeit
import scipy
import numpy as np
from math import sqrt
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score, \
    confusion_matrix
from data_merge import data_load
from networks.model_bagca import DGMM_DTI
from thop import profile
import warnings
from prettytable import PrettyTable
import random

torch.multiprocessing.set_start_method('spawn')

class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay) # AdamW
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset, p_LMs, d_LMs, epoch):
        np.random.shuffle(dataset)
        N = len(dataset)

        loss_total = 0
        i = 0
        # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=[0, 1])
        self.optimizer.zero_grad()

        molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, molecule_edge, protein, sequence, smile, label, source = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            molecule_edges.append(molecule_edge)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)
            sources.append(source)

            if i % self.batch_size == 0 or i == N:
                if len(molecule_words) != 1:
                    molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, sources = pack1(molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, p_LMs, d_LMs, device, sources)
                    data = (molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, sources)
                    loss = self.model(data, epoch)  #.mean()
                    # flops, params = profile(self.model, (data,epoch))
                    # print("!!!", flops/1000**3, params/1000**2)
                    # loss = loss / self.batch
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], [], []
                else:
                    molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], [], []
            else:
                continue


            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                # self.schedule.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
            # loss_total2 += loss3.item()

        return loss_total

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset, p_LMs, d_LMs):
        N = len(dataset)
        T, S, Y, S2, Y2, S3, Y3 = [], [], [], [], [], [], []
        i = 0
        molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, molecule_edge, protein, sequence, smile, label = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            molecule_edges.append(molecule_edge)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                # print(words[0])
                molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, _ = pack1(molecule_words, molecule_atoms,
                                                                                       molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, p_LMs, d_LMs,
                                                                                       device)
                # print(words.shape)
                data = (molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins, sequences, smiles, labels, _)
                # print(self.model(data, train=False))
                correct_labels, ys1, ys2, ys3 = self.model(data, train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                ys1 = ys1.to('cpu').data.numpy()
                ys2 = ys2.to('cpu').data.numpy()
                ys3 = ys3.to('cpu').data.numpy()
                predicted_labels1 = list(map(lambda x: np.argmax(x), ys1))
                predicted_scores1 = list(map(lambda x: x[1], ys1))
                predicted_labels2 = list(map(lambda x: np.argmax(x), ys2))
                predicted_scores2 = list(map(lambda x: x[1], ys2))
                predicted_labels3 = list(map(lambda x: np.argmax(x), ys3))
                predicted_scores3 = list(map(lambda x: x[1], ys3))

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    Y.append(predicted_labels1[j])
                    S.append(predicted_scores1[j])
                    Y2.append(predicted_labels2[j])
                    S2.append(predicted_scores2[j])
                    Y3.append(predicted_labels3[j])
                    S3.append(predicted_scores3[j])

                molecule_words, molecule_atoms, molecule_adjs, molecule_edges, proteins,  sequences, smiles, labels = [], [], [], [], [], [], [], []
            else:
                continue

        AUC = roc_auc_score(T, S)
        Accuracy = accuracy_score(T, Y)
        AUC2 = roc_auc_score(T, S2)
        AUC3 = roc_auc_score(T, S3)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)

        PRC = auc(fpr, tpr)
        f1 = 2 * precision * recall / (recall + precision + 0.00001)
        cm1 = confusion_matrix(T,Y)
        specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        return AUC, precision, PRC, AUC2, AUC3, recall,Accuracy,f1,sensitivity,specificity

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_select = "BIO_to_BIO1"
    refine = "start"
    iteration = 100
    decay_interval = 5
    batch_size = 16 # 16
    lr = 5e-4 # 5e-4
    weight_decay = 0.07
    lr_decay = 0.5
    layer_gnn = 3

    drop = 0.0
    setting = "1"
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_val, dataset_test, p_LMs, d_LMs = data_load(data_select, device)
    # setup_seed(2000)
    set_seed(2001)

    model = DBCA_DTI(layer_gnn=layer_gnn, device=device, dropout=drop).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[1], output_device=1)
    # model = model.module.to(torch.device('cpu'))
    trainer = Trainer(model, batch_size, lr, weight_decay)
    tester = Tester(model, batch_size)

    """Output files."""
    file_AUCs = 'output/result/' + setting + '.txt'
    file_model = 'output/model/'+ f'{data_select}' + setting + '.pth'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'AUC_test\tAUPR_test\tAcc\tPrecision_test\trecall\tf1\tsensitivity\tspecificity\tAUC_LM\tAUC_Sty\t')

    output_dir = 'output/result/' + f'{data_select}/' +f'{refine}'
    file_AUCs = os.path.join(output_dir, 'AUCs--H_to_H.txt')

    # 确保目标目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    auc1 = 0
    for epoch in range(1, iteration):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, p_LMs, d_LMs, epoch)
        AUC_test, precision_test, PRC_test, AUC2, AUC3,recall_test,Acc,f1,sensitivity,specificity = tester.test(dataset_val, p_LMs, d_LMs)

        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch, time, loss_train,
                AUC_test, PRC_test,Acc,precision_test,  recall_test,f1,sensitivity,specificity,AUC2, AUC3]
        tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        if auc1 < AUC_test:
            auc1 = AUC_test
            epoch1 =  epoch
            print("best epoch:", epoch1)
            tester.save_model(model, file_model)

    model1 = torch.load(file_model)
    model.load_state_dict(model1)
    print("test: !!!!") 
    AUC_test, precision_test, PRC_test, AUC2, AUC3,recall_test,Acc,f1,sensitivity,specificity = tester.test(dataset_test, p_LMs, d_LMs)
    # AUCs = [epoch1, AUC_test, AUC2, AUC3, precision_test, PRC_test, recall_test,Acc,f1,sensitivity,specificity]
    # AUCs = [epoch1,AUC_test, PRC_test, precision_test, PRC_test, AUC2, AUC3, recall_test, Acc, f1, sensitivity, specificity]

    AUCs = [epoch, time, loss_train,
            AUC_test, PRC_test, Acc, precision_test, recall_test, f1, sensitivity, specificity, AUC2, AUC3]
    print('\t'.join(map(str,AUCs)))

    # result = output_dir + '/results.txt'
    test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "Accuracy", "Precision","Recall",  "F1", "Sensitivity", "Specificity"]
    test_table = PrettyTable(test_metric_header)
    float2str = lambda x: '%0.4f' % x if isinstance(x, (float, np.float32, np.float64)) else str(x)
    test_lst = ["epoch " + str(epoch1)] + list(map(float2str,[AUC_test, PRC_test, Acc, precision_test,recall_test,
                                                                              f1, sensitivity,
                                                                             specificity]))
    test_table.add_row(test_lst)

    test_prettytable_file = os.path.join(output_dir, "test_markdowntable.txt")
    with open(test_prettytable_file, 'w') as fp:
        fp.write(test_table.get_string())
