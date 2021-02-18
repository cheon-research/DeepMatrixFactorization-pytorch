"""
Implementation for "Deep Matrix Factorization Models for Recommender Systems"
IJCAI'17: Proceedings of the 26th International Joint Conference on Artificial Intelligence (August 2017)

https://www.ijcai.org/Proceedings/2017/447

by Sangjin Cheon (cheon.research @ gmail.com)
University of Seoul, Korea
"""

import torch
import torch.optim as optim
import numpy as np

import time

from model import DMF
import data_utils
from functions import *


def run(dataset, layers, n_negs, gpu='0'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print("##### {} Negative Samples experiment on {} {}".format(n_negs, dataset, layers))
    
    learning_rate = 0.0001
    batch_size = 256

    if torch.cuda.is_available():
        device = torch.device('cuda')
        FloatTensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        FloatTensor = torch.FloatTensor
    manualSeed = 706
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print('CUDA Available:', torch.cuda.is_available())

    file_name = 'output/' + dataset + '_DMF_eps_' + str(layers) + '_n_' + str(n_negs) + '.txt'
    output = open(file_name, 'w')

    # Datasets
    user_matrix, item_matrix, train_u, train_i, train_r, neg_candidates, u_cnt, user_rating_max = data_utils.load_train_data(dataset)
    if dataset == 'ml1m':
        epochs = 100
        eval_batch_size = 100 * 151
        test_users, test_items = data_utils.load_test_ml1m()
    elif dataset == 'ml100k':
        epochs = 100
        eval_batch_size = 100 * 41
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'yelp':
        epochs = 50
        eval_batch_size = 100 * 81
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'amusic':
        epochs = 100
        eval_batch_size = 100 * 3
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'agames':
        epochs = 100
        eval_batch_size = 100 * 34
        test_users, test_items = data_utils.load_test_data(dataset)
    n_users, n_items = user_matrix.shape[0], user_matrix.shape[1]

    user_array = user_matrix.toarray()
    item_array = item_matrix.toarray()
    user_idxlist, item_idxlist = list(range(n_users)), list(range(n_items))

    # Model
    model = DMF(layers, n_users, n_items).to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_hr = 0.0
    for epoch in range(epochs):
        # Train
        model.train()  # Enable dropout (if have).

        # Negative Sampling
        new_users, new_items, new_labels = dmf_negative_sampling(train_u, train_i, train_r, u_cnt, neg_candidates, n_negs)

        idxlist = np.array(range(len(new_users)))
        np.random.shuffle(idxlist)
        epoch_loss = .0

        start_time = time.time()
        for batch_idx, start_idx in enumerate(range(0, len(idxlist), batch_size)):
            end_idx = min(start_idx + batch_size, len(idxlist))
            idx = idxlist[start_idx:end_idx]

            u_ids = new_users.take(idx)
            i_ids = new_items.take(idx)
            
            users = FloatTensor(user_array.take(u_ids, axis=0))
            items = FloatTensor(item_array.take(i_ids, axis=0))
            labels = FloatTensor(new_labels.take(idx))

            rating_max = FloatTensor(user_rating_max.take(u_ids, axis=0))
            Y_ui = labels / rating_max  # for Normalized BCE

            optimizer.zero_grad()

            preds = model(users, items)
            try:
                loss = loss_function(preds, Y_ui)
                epoch_loss += loss.item()
            except:
                print(preds)
                print(loss)
                exit()

            
            loss.backward()
            optimizer.step()
        train_time = time.time() - start_time

        # Evaluate
        model.eval()
        HR, NDCG = [], []

        time_E = time.time()
        for start_idx in range(0, len(test_users), eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, len(test_users))
            u_ids = test_users[start_idx:end_idx]
            i_ids = test_items[start_idx:end_idx]

            users = FloatTensor(user_array.take(u_ids, axis=0))
            items = FloatTensor(item_array.take(i_ids, axis=0))

            preds = model(users, items).detach().cpu()

            e_batch_size = eval_batch_size // 100  # faster eval
            preds = torch.chunk(preds, e_batch_size)
            chunked_items = torch.chunk(torch.IntTensor(i_ids), e_batch_size)

            for i, pred in enumerate(preds):
                _, indices = torch.topk(pred, 10)
                recommends = torch.take(chunked_items[i], indices).numpy().tolist()

                gt_item = chunked_items[i][0].item()
                HR.append(hit(gt_item, recommends))
                NDCG.append(ndcg(gt_item, recommends))

        eval_time = time.time() - time_E
        
        text = '[Epoch {:03d}] Loss: {:.6f}'.format(epoch, epoch_loss / (batch_idx + 1)) + '\ttrain: ' + time.strftime('%M: %S', time.gmtime(train_time)) + '\tHR: {:.4f}\tNDCG: {:.4f}\n'.format(np.mean(HR), np.mean(NDCG))
        if epoch % 10 == 0:
          print(text[:-1])
        output.write(text)

        if np.mean(HR) > best_hr:
            best_hr, best_ndcg, best_epoch = np.mean(HR), np.mean(NDCG), epoch

    result = '{} Best epoch {:02d}: HR = {:.4f}, NDCG = {:.4f}\n'.format(layers, best_epoch, best_hr, best_ndcg)
    print(result[:-1])
    output.write(result)
    output.close()

if __name__ == "__main__":
    layers = [512, 64]
    run("amusic", layers, 5, '7')
