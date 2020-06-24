import pickle
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
from torch.autograd import Variable

def generate_input_history(data_neural, mode, candidate=None):
    data = {}
    traj_idx = {}

    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        traj_id = data_neural[u][mode]

        data[u] = {} 

        for c, i in enumerate(traj_id):

            # if mode=='train' and c==0:
            #     continue
            session = sessions[i]
            trace = {}
            # history = []

            # if mode == 'test':
            #     all_train_id = data_neural[u]['train']
            #     for j in all_train_id:
            #         history.extend([(s[0], s[1]) for s in sessions[j]])
            # for j in range(c):
            #     history.extend([(s[0], s[1]) for s in sessions[traj_id[j]]])
            
            # history_count = [1]
            # history_tim = [x[1] for x in history]
            # last_t = history_tim[0]
            # count = 1
            # for t in history_tim[1:]:
            #     if t == last_t:
            #         count += 1
            #     else:
            #         history_count[-1] = count
            #         history_count.append(1)
            #         last_t = t
            #         count = 1
            
            # history_tim = np.reshape(np.array([x[1] for x in history]), (-1))
            # history_loc = np.reshape(np.array([x[0] for x in history]), (-1))
            # trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            # trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            # trace['history_count'] = history_count

            target = np.array([s[0] for s in session[1:]])

            # loc_tim = history
            loc_tim = []
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (-1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (-1))
            # trace['loc'] = Variable(torch.LongTensor(loc_np))
            # trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['target'] = Variable(torch.LongTensor(target))
            trace['loc'] = loc_np
            trace['tim'] = tim_np
            trace['target'] = target

            data[u][i] = trace
        traj_idx[u] = traj_id
        
    # data[u][i]: {'history_loc', 'hitory_tim', 'history_count',
    #        'loc', 'tim', 'target'}    
    # traj_idx: {u: train_id or test_id}
    return data, traj_idx


            

def get_random_user_traj(traj_id):
    len_trajs = len(traj_id)
    return random.randrange(len_trajs)

def generate_input(data_neural):

    users = list(data_neural.keys())
    num_users = len(users)
    traj_input_list = []

    for u in data_neural.keys():

        train_ids = data_neural[u]['train']
        for traj_id in train_ids[:-1]:

            prob = random.random()
            if prob > 0.5:
               traj_input_list.append(((u, traj_id), (u, traj_id+1), 1))   
            elif prob <= 0.5:
                rand_user = random.randrange(num_users)
                rand_id = get_random_user_traj(data_neural[rand_user]['train'])
                # rand_user = u
                # rand_id = get_random_user_traj(data_neural[rand_user]['train'])
                traj_input_list.append(((u, traj_id), (rand_user, rand_id), 0))

    return traj_input_list

def generate_input2(data_neural):
    users = list(data_neural.keys())
    num_users = len(users)
    train_input_list = []
    test_input_list = []

    for u in data_neural.keys():
        train_ids = data_neural[u]['train']
        test_ids = data_neural[u]['test']
        for traj_id in train_ids:
            train_input_list.append((u, traj_id))
        for traj_id in test_ids:
            test_input_list.append((u, traj_id))

    return train_input_list, test_input_list


def get_traj(data, data_neural):

    u, id = data

    sessions = data_neural[u]['sessions']
    session = sessions[id]

    loc_tim = []
    loc_tim.extend([(s[0], s[1]) for s in session])
    loc = [s[0] for s in loc_tim]
    tim = [s[1] for s in loc_tim]

    return loc, tim
    # return loc_np, tim_np


class Predict_Traj_Dataset(Dataset):

    def __init__(self, data_neural, traj_list, vid_list):
        super(Predict_Traj_Dataset, self).__init__()

        self.data = data_neural
        self.trajs = traj_list
        self.vid_list = vid_list
        self.seq_len = 11

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        # self.trajs: [(u, id) ...]
        (u1, id1) = self.trajs[idx]

        loc, tim = get_traj((u1, id1), self.data)
        tim = [i+1 for i in tim]

        loc1 = [self.vid_list['pad'][0]] + loc[:-1]

        tim1 = [self.vid_list['pad'][0]] + tim[:-1]

        loc_label = [self.vid_list['pad'][0]] + loc[1:]
        time_label = [self.vid_list['pad'][0]] + tim[1:]

        segment_label = ([1 for _ in range(len(loc1))])

        bert_input = (loc1)[:self.seq_len]
        bert_label = (loc_label)[:self.seq_len]
        time_input = (tim1)[:self.seq_len]
        time_label = (time_label)[:self.seq_len]

        padding = [self.vid_list['pad'][0] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding), time_input.extend(padding), time_label.extend(padding)
        # print(bert_input, bert_label, segment_label, is_Next)
        bert_input, bert_label, segment_label, time_input, time_label = \
                    np.array(bert_input), np.array(bert_label), np.array(segment_label), np.array(time_input), np.array(time_label)

        return bert_input, bert_label, segment_label, time_input, time_label, u1

    

# 读取数据
    # foursquare_dataset = {
    #     'data_neural': self.data_neural,
    #     'vid_list': self.vid_list,   'uid_list': self.uid_list,
    #     'data_filter': self.data_filter,
    #     'vid_lookup': self.vid_list_lookup }

# dataset_4qs = pickle.load(open('./data/tweets-cikm.txtfoursquare.pk', 'rb'))

# print("pad", dataset_4qs['vid_list']['pad'])
# print("cls", dataset_4qs['vid_list']['cls'])
# print("sep", dataset_4qs['vid_list']['sep'])
# print("unk", dataset_4qs['vid_list']['unk'])
# print("mask", dataset_4qs['vid_list']['mask'])

    # data[u][i]: {'history_loc', 'hitory_tim', 'history_count',
    #              'loc', 'tim', 'target'}    
#     # traj_idx: {u: train_id or test_id}
# train_data, train_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="train")
# test_data, test_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="test")

# train_queue = generate_queue(train_traj_idx, 'normal', 'tr')
# len_queue = len(train_queue)
# len_batch = int(np.ceil((len_queue/batch_size)))
# train_queue = [[train_queue.popleft() for _ in range(min(batch_size, len(train_queue)))] for k in range(len_batch)]


# for i, batch_queue in enumerate(train_queue):

#     max_place = max([len(train_data[u][id]['loc']) for u,id in batch_queue])

#     place = torch.cat([pad_tensor(train_data[u][id]['loc']+1,max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()
#     time = torch.cat([pad_tensor(train_data[u][id]['tim']+1,max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()
#     target = torch.cat([pad_tensor(train_data[u][id]['target']+1,max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0)

#     max_history_place = max([len(train_data[u][id]['history_loc']) for u,id in batch_queue])

#     history_place = torch.cat([pad_tensor(train_data[u][id]['history_loc']+1,max_history_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()
#     history_time = torch.cat([pad_tensor(train_data[u][id]['history_tim']+1,max_history_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()

#     target = target.to(device).long()
#     target = target.contiguous().view(-1)

# print(train_traj_idx)
# print(train_queue)


# print(dataset_4qs['data_neural'].keys())



# traj_input_list = generate_input(dataset_4qs['data_neural'])

# # is_next是否考虑换为同一人的不同轨迹
# train_dataset = Bert_Traj_Dataset(data_neural=dataset_4qs['data_neural'], traj_list=traj_input_list, vid_list=dataset_4qs['vid_list'])
# trainloader = DataLoader(train_dataset, batch_size=2,shuffle=True)












