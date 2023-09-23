# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-14 20:57
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-14 20:57
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from torch.optim import SGD, Adam

from aggregation.synchronous import SLMFed_syn, FedAVG, FedLAMA, E3CS, SoteriaFL
from server.standalone import SLMFedStageServer, NormalStageServer, FedProfStageServer, OortStageServer, \
    E3CSStageServer, FedBalancerStageServer, SoteriaFLStageServer
from trainer.direct import DirectTrainer,ProxTrainer,FedNTDTrainer,SoteriaFLTrainer
from client.standalone import StandAloneClient
from utils.random import random_seed

device = torch.device('cuda:0')

from model.MNIST.CNN import create_cnn_for_mnist
from model.FMNIST.CNN import create_cnn_for_fmnist
from model.CIFAR10.CNN import create_cnn_for_cifar10
from model.GERMANTS.CNN import create_cnn_for_germants

cifar10_path = './model/CIFAR10/cifar10.pth'
fmnist_path = './model/FMNIST/fmnist.pth'
mnist_path = './model/MNIST/mnist.pth'
germants_path = './model/GERMANTS/germants.pth'

# Model_func,lr,model_path,batchsize,maxround,optim,target_pnew,target_ue,target_acc,convergence
dataset_dict = {
    'mnist': [create_cnn_for_mnist, 0.05, mnist_path, 32, 10, SGD, 0.41, 0.7, 1, 0.01],
    'fmnist': [create_cnn_for_fmnist, 0.05, fmnist_path, 32, 20, SGD, 0.398, 0.65, 0.9, 0.02],
    'cifar10': [create_cnn_for_cifar10, 0.005, cifar10_path, 32, 55, SGD, 0.4, 0.55, 0.6, 0.05],
    'germants': [create_cnn_for_germants, 0.1, germants_path, 32, 35, SGD, 0.4, 0.7, 0.7, 0.02],
}

# Trainer,aggregation
experiment_method = {
    'SLMFed': [DirectTrainer,SLMFed_syn,SLMFedStageServer,'random_perc'],
    'SLMFed_random': [DirectTrainer,SLMFed_syn,SLMFedStageServer,'random'],
    'SLMFed_rank': [DirectTrainer,SLMFed_syn,SLMFedStageServer,'rank_perc'],
    'FedAvg': [DirectTrainer,FedAVG,NormalStageServer,'random'],
    'FedProx': [ProxTrainer,FedAVG,NormalStageServer,'random'],
    'NOMA': [DirectTrainer,FedAVG,NormalStageServer,'NOMA'],
    'FedProf': [DirectTrainer,FedAVG,FedProfStageServer,'FedProf'],
    'FedLAMA': [DirectTrainer,FedLAMA,NormalStageServer,'random'],
    'FedNTD': [FedNTDTrainer,FedAVG,NormalStageServer,'random'],
    'SoteriaFL': [SoteriaFLTrainer,SoteriaFL,SoteriaFLStageServer,'random'],
    'Oort': [DirectTrainer,FedAVG,OortStageServer,'Oort'],
    'E3CS': [DirectTrainer,E3CS,E3CSStageServer,'E3CS'],
    'FedBalancer': [DirectTrainer,FedAVG,FedBalancerStageServer,'FedBalancer'],
}

client_train_epoch = 5
client_num = 200
client_init_num = 100

data_incre_min_perc, data_incre_prob = 0, 0.5
client_incre_min_perc, client_incre_max_perc, client_incre_prob = 0.01, 0.015, 1.1
data_incre_max_perc_alter = [0.01,0.1]

choose_item = ['random_perc','rank_perc','random']
select_percentage = 0.25

stage_num = 5


def get_clients_id(min_id, max_id):
    result = []
    for i in range(min_id, max_id + 1):
        if i < 10:
            result.append('00' + str(i))
        elif 10 <= i < 100:
            result.append('0' + str(i))
        else:
            result.append(str(i))
    return result


def adjust_channel(data):
    return np.transpose(data, (0, 3, 1, 2))


def main():
    with open("./exp.txt", "a", encoding='utf-8') as f:
        for dataset_name in ['mnist','fmnist', 'germants', 'cifar10']:
            for method_name in experiment_method:
                Model_func = dataset_dict[dataset_name][0]
                lr = dataset_dict[dataset_name][1]
                model_path = dataset_dict[dataset_name][2]
                batch_size = dataset_dict[dataset_name][3]
                max_round = dataset_dict[dataset_name][4]
                optim = dataset_dict[dataset_name][5]
                target_pnew = dataset_dict[dataset_name][6]
                target_ue = dataset_dict[dataset_name][7]
                target_pnew = 1.1
                target_ue = -1
                target_acc = dataset_dict[dataset_name][8] * 100
                convergence = dataset_dict[dataset_name][9]
                base_acc = target_acc - 20
                target_acc=100
                data_dir = './data/' + dataset_name + '_noniid/'
                for data_incre_max_perc in data_incre_max_perc_alter:
                    Trainer = experiment_method[method_name][0]
                    Aggregation = experiment_method[method_name][1]
                    Server_object = experiment_method[method_name][2]
                    choose = experiment_method[method_name][3]
                    if True:
                        stage_id = 1
                        print(
                            'model:' + str(dataset_name) + '\n' +
                            'method:' + str(method_name) + '\n' +
                            'lr:' + str(lr) + '\n' +
                            'incre_max:' + str(data_incre_max_perc) + '\n' +
                            'choose:' + str(choose) + '\n' +
                            'select_percentage:' + str(select_percentage) + '\n'
                        )
                        f.writelines(
                            'model:' + str(dataset_name) + '\n' +
                            'method:' + str(method_name) + '\n' +
                            'lr:' + str(lr) + '\n' +
                            'incre_max:' + str(data_incre_max_perc) + '\n' +
                            'choose:' + str(choose) + '\n' +
                            'select_percentage:' + str(select_percentage) + '\n'
                        )
                        f.flush()

                        random_seed(1)

                        data_stimulus = adjust_channel(
                            np.load('./data/data_stimulus/' + dataset_name + '_x_stimulus.npy'))

                        client_train_id_list = get_clients_id(1, 200)
                        client_train_list = []
                        for c_id in client_train_id_list:
                            client_id = 'client_' + str(c_id)
                            data = adjust_channel(np.load(
                                data_dir + client_id + '_x_train.npy'))
                            label = np.load(
                                data_dir + client_id + '_y_train.npy')
                            model = Model_func()
                            trainer_temp = Trainer(
                                model=model,
                                optimizer=optim(model.parameters(), lr=lr),
                                learning_rate=lr,
                                loss=torch.nn.CrossEntropyLoss(),
                                batch_size=batch_size,
                                train_epoch=client_train_epoch,
                                device=device,
                                cal_cost_base_option=False
                            )
                            client_temp = StandAloneClient(
                                client_name=client_id,
                                train_data=data,
                                train_label=label,
                                trainer=trainer_temp,
                                data_incre_min_perc=data_incre_min_perc,
                                data_incre_max_perc=data_incre_max_perc,
                                data_incre_prob=data_incre_prob,
                                incremental=True,
                                initial_rate=0.5,
                            )
                            client_train_list.append(client_temp)

                        model = Model_func()
                        model.load_state_dict(torch.load(model_path))
                        client_now = client_train_list[0:client_init_num]
                        client_rest = client_train_list[client_init_num:]
                        eval_client= client_train_list
                        server = Server_object(
                            aggregation=Aggregation(),
                            model=model,
                            select=choose,
                            select_percentage=select_percentage,
                            base_acc=base_acc,
                            client_incre_min_perc=client_incre_min_perc,
                            client_incre_max_perc=client_incre_max_perc,
                            client_incre_prob=client_incre_prob,
                            target_pnew=target_pnew,
                            target_ue=target_ue,
                            stage_total=stage_num,
                            data_stimulus=data_stimulus,
                            loss=torch.nn.CrossEntropyLoss(),
                            eval_target=target_acc,
                            convergence=convergence,
                            convergence_round=int(max_round * 4/5),
                            server_name='Server1',
                            eval_client=eval_client,
                            batch_size=batch_size,
                            device=device,
                            round_increment=5,
                            record_file=f,
                            client_now=client_now,
                            client_rest=client_rest
                        )
                        server.start_task(maxround=max_round)

        f.close()


if __name__ == '__main__':
    main()
