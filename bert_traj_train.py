import time
import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_generate import Predict_Traj_Dataset, generate_input, generate_input2
from bert_language_model import BertLM, Predict_Model
from bert_traj_model import Bert_Traj_Model
from optimizer import Trans_Optim
from eval_traj import cal_loss_performance, get_acc

def save_model(Epoch, model, file_path="./pretrain/", Predict=False):
    if Predict:
        output_path = file_path + "Predict_model_trained_ep%d.pth" % Epoch
    else:
        output_path = file_path + "Pretrained_ep%d.pth" % Epoch
    # bert_output_path = file_path + "bert_trained_ep%d.pth" % Epoch
    torch.save(model.state_dict(), output_path)
    print("EP:%d Model Saved on:" % Epoch, output_path)
    if not Predict:
        model.save_bert(Epoch)
    print()
    return True

def predict_train_epoch(epoch, model, train_traj_dataloader, optimizer, device):

    model.train()
    desc= ' -(Train)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))

    for i, data in enumerate(train_traj_dataloader):

        bert_input, bert_label, _, time_input, time_label, user = data[0].to(device).long(), \
                                                                        data[1].to(device).long().contiguous().view(-1), \
                                                                        data[2].to(device).long(), \
                                                                        data[3].to(device).long(), \
                                                                        data[4].to(device).long().contiguous().view(-1), \
                                                                        data[5].to(device).long()
                                                                        #  data[1].to(device).long().contiguous().view(-1), \

        place_logit, time_logit = model(bert_input, time_input, user)

        loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, logit2=time_logit, label1=bert_label, label2=time_label, Predict=True)
        eva_metric = get_acc(bert_label, place_logit, eva_metric)

        total_loss += loss.item()
        iter_100_loss += loss.item()
        avg_loss = total_loss/(i+1)

        total_loc += n_loc
        iter_100_loc += n_loc
        total_cor_loc += n_cor_loc
        iter_100_cor_loc += n_cor_loc
        iter_100_cor_time += n_cor_time
        avg_acc = 100.*total_cor_loc/total_loc

        if i % 100 == 0:
            try:
                if n_loc==0 and n_cor_loc==0:
                    n_loc = 1
                print("{} epoch: {:_>2d} | iter: {:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f} | time_acc: {:<4.4f} %".format(
                    desc, epoch, i, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr(), 100.*iter_100_cor_time/iter_100_loc))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
            except Exception as e:
                print(e)
                exit()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

    return avg_loss, avg_acc, eva_metric/total_loc

def predict_valid_epoch(epoch, model, valid_traj_dataloader, optimizer, device):

    model.eval()
    desc= ' -(Valid)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time= 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))
    
    with torch.no_grad():
        for i, data in enumerate(valid_traj_dataloader):
        # bert_input, bert_label, segment_label, is_Next
           
            bert_input, bert_label, _, time_input, time_label, user = data[0].to(device).long(), \
                                                                        data[1].to(device).long().contiguous().view(-1), \
                                                                        data[2].to(device).long(), \
                                                                        data[3].to(device).long(), \
                                                                        data[4].to(device).long().contiguous().view(-1), \
                                                                        data[5].to(device).long()
            place_logit, time_logit = model(bert_input, time_input, user)

            loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, logit2=time_logit, label1=bert_label, label2=time_label, Predict=True)
            eva_metric = get_acc(bert_label, place_logit, eva_metric)

            total_loss += loss.item()
            iter_100_loss += loss.item()
            avg_loss = total_loss/(i+1)

            total_loc += n_loc
            iter_100_loc += n_loc
            total_cor_loc += n_cor_loc
            iter_100_cor_loc += n_cor_loc
            iter_100_cor_time += n_cor_time
            avg_acc = 100.*total_cor_loc/total_loc
            
            if i % 100 == 0:
                print("{} epoch: {:_>2d} | iter: {:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f} | time_acc: {:<4.4f} %".format(
                    desc, epoch, i, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr(), 100.*iter_100_cor_time/iter_100_loc))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0

    return avg_loss, avg_acc, eva_metric/total_loc


def run(epoch, model, optimizer, device, train_traj_dataloader, valid_traj_dataloader=False, Predict=False, log=None):

    # with SummaryWriter() as writer:

    log_train_file, log_valid_file = None, None

    if log:
        log_train_file = log + '.train.log'
        log_valid_file = log + '.valid.log'
        # print('[Info] Training performance will be written to file: {} and {}'.format(
        #     log_train_file, log_valid_file))
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                log_vf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        
    for epoch_i in range(1, epoch+1):
        if Predict:
            # train_avg_loss, train_acc, train_metric = predict_train_epoch(epoch_i, model, train_traj_dataloader, optimizer, device)
            valid_avg_loss, valid_acc, valid_metric = predict_valid_epoch(epoch_i, model, valid_traj_dataloader, optimizer, device)
            print('-'*150)
            print(" --Train--  Epoch: {}/{}  Train_avg_loss: {:<10.7f} Train_acc: {:<4.4f}".format(epoch_i, epoch, train_avg_loss, train_acc))
            print(" --Valid--  Epoch: {}/{}  Valid_avg_loss: {:<10.7f} Valid_acc: {:<4.4f}".format(epoch_i, epoch, valid_avg_loss, valid_acc))
            print('-'*150)
            print(" --Train--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, train_metric[0][0], train_metric[1][0], train_metric[2][0], train_metric[3][0], train_metric[4][0], train_metric[5][0]))
            print(" --Valid--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, valid_metric[0][0], valid_metric[1][0], valid_metric[2][0], valid_metric[3][0], valid_metric[4][0], valid_metric[5][0]))
            print('-'*150)
            # exit()
            # *********************************************************************
        else:
            # train_avg_loss, train_ns_acc, train_ml_acc = train_epoch(epoch_i, model, train_traj_dataloader, optimizer, device)
            # print(" --Train--  epoch: {}/{}".format(epoch_i, epoch)," Train_avg_loss: {} Train_ns_acc: {} Train_ml_acc: {}".format( \
            #     train_avg_loss, train_ns_acc, train_ml_acc))
            # valid_total_loss, valid_epoch_acc = eval_epoch(
            #     model, valid_data, valid_traj_idx, device, epoch_i, batch_size)
            # print(" --Valid--  epoch: {}/{}".format(epoch_i, epoch)," Valid_loss: ", valid_total_loss, " Valid_acc:", valid_epoch_acc, "%")
            print()
        
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(" --Train--  Epoch: {}/{}  Train_avg_loss: {} Train_acc: {}\n".format(epoch_i, epoch, train_avg_loss, train_acc))
                log_vf.write(" --Valid--  Epoch: {}/{}  Valid_avg_loss: {} Valid_acc: {}\n".format(epoch_i, epoch, valid_avg_loss, valid_acc)) 
        
        if epoch_i % 3==0:
            save_model(epoch_i, model, Predict=Predict)
            # writer.add_scalars("Loss", {"Train": train_total_loss, "Valid": valid_total_loss}, epoch_i)
            # writer.add_scalars("Acc", {"Train": train_epoch_acc, "Valid": valid_epoch_acc}, epoch_i)
            # writer.add_scalars("Lr", {"Train": optimizer._print_lr()}, epoch_i)


def main(Epoch=5, Pretrain=False, Predict=False, Batch_size=2, log=None):

    print('*'*150)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Get Dataset")
    dataset_4qs = pickle.load(open('./data/tweets-cikm.txtfoursquare.pk', 'rb'))
    print("User number: ", len(dataset_4qs['uid_list']))
    print("Generate Traj_list")
    if Predict:
        train_input_list, test_input_list = generate_input2(dataset_4qs['data_neural'])
    else:
        traj_input_list = generate_input(dataset_4qs['data_neural'])
    print("Generate Train Dataset")
    if Predict:
        train_dataset = Predict_Traj_Dataset(data_neural=dataset_4qs['data_neural'], traj_list=train_input_list, vid_list=dataset_4qs['vid_list'])
        valid_dataset = Predict_Traj_Dataset(data_neural=dataset_4qs['data_neural'], traj_list=test_input_list, vid_list=dataset_4qs['vid_list'])
        print("Generate Train Dataloader")
        train_traj_dataloader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
        valid_traj_dataloader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=True)
    else:
        train_dataset = Bert_Traj_Dataset(data_neural=dataset_4qs['data_neural'], traj_list=traj_input_list, vid_list=dataset_4qs['vid_list'])
        print("Generate Train Dataloader")
        train_traj_dataloader = DataLoader(train_dataset, batch_size=Batch_size,shuffle=True)

    if Predict:
        if Pretrain:
            print("Loaded Pretrained Bert")      
            Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1)
            Bert.load_state_dict(torch.load('./pretrain/bert_trained_ep14.pth')) 
        else: 
            print("Create New Bert")      
            Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1)
        print("Get Predict Model")
        model = Predict_Model(Bert, token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.2) 
        # model.load_state_dict(torch.load('./pretrain/Predict_model_trained_ep45.pth'))
        model = model.to(device)
    else:
        if Pretrain:
            print("Get Pretrained Model")
            Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1)
            model = BertLM(token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1)
            model.load_state_dict(torch.load('./pretrain/bert_trained'))   
            model = model.to(device)
        else:
            print("Initialite Model")
            Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1)
            model = BertLM(Bert_Traj_Model=Bert, token_size=len(dataset_4qs['vid_list']), head_n=12, d_model=512, N_layers=12, dropout=0.1).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    optimizer = Trans_Optim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        init_lr=2, d_model=512, n_warmup_steps=10000)
    print('*'*150)
    print('-'*65 + "  START TRAIN  " + '-'*65)
    run(Epoch, model, optimizer, device, train_traj_dataloader, valid_traj_dataloader=valid_traj_dataloader,Predict=Predict, log=log)

if __name__ == "__main__":
    main(Epoch=50, Pretrain=False, Batch_size=4, Predict=True, log='predict')
    pass





