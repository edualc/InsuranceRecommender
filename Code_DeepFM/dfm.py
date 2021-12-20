import os
import time
import argparse
import itertools
import copy
from math import log
from datetime import datetime
import csv
from tqdm import tqdm

import scipy
import scipy.sparse
from scipy.sparse import csr_matrix, save_npz, load_npz
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import tensorflow as tf
from libreco.data import random_split, DatasetFeat, DatasetPure, split_by_ratio
from libreco.data.data_generator import DataGenFeat
from libreco.algorithms import DeepFM  # feat data
from libreco.algorithms.base import Base, TfMixin
from libreco.evaluation import evaluate
from libreco.evaluation.computation import compute_probs
from libreco.evaluation.evaluate import EvalMixin

import wandb

# Max number of interactions per evaluation chunk
# (~= dense matrix dimensions)
NUMPY_ARRAY_CHUNKING_LIMIT = 2 ** 18
DEBUG = False

DATASET_LOCATIONS = {
    'insurance': 'data/insurance',
    'ml1m_min6': 'data/ml-1m/min6',
    'ml1m_max5': 'data/ml-1m/max5_pointwisesplit_oldestperuser_notscaleprice',
    'yoochoose': 'data/yoochoose',
    'yoochoose_small': 'data/yoochoose_small',
    'retailrocket': 'data/retailrocket/retailrocket'
}

ROW_NAME_BY_DATASET = {
    'insurance': 'userId',
    'ml1m_min6': 'customer_id',
    'ml1m_max5': 'customer_id',
    'yoochoose': 'Session ID',
    'yoochoose_small': 'Session ID',
    'retailrocket': 'visitorid'
}

COL_NAME_BY_DATASET = {
    'insurance': 'itemId',
    'ml1m_min6': 'product_id',
    'ml1m_max5': 'product_id',
    'yoochoose': 'Item ID',
    'yoochoose_small': 'Item ID',
    'retailrocket': 'itemid'
}

PRICE_COL_NAME_BY_DATASET = {
    'insurance': 'price',
    'ml1m_min6': 'Price',
    'ml1m_max5': 'Price',
    'yoochoose': 'Price',
    'yoochoose_small': 'Price',
    'retailrocket': 'Price'
}


# lehl@2021-11-29: 
# Generate full user-item matrix prediction, see for comparison:
# https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/ncf.py#L167
#
# In case that user_rows are provided (the indices),
# only use those for the prediction
#
def predict(model, user_rows=None):
    user_ids = np.array(list(map(lambda x: np.full(model.n_items, x), user_rows))).flatten()
    item_ids = np.tile(np.arange(model.n_items), len(user_rows))

    feed_dict = model._get_feed_dict(user_ids, item_ids, None, None, None, False)
    pred = model.sess.run(model.output, feed_dict)

    pred = pred.reshape(len(user_rows), model.n_items)
    pred = 1 / (1 + np.exp(-pred))

    return pred

def run_dfm(train, test, prices, args):
    folder_path = '/'.join([DATASET_LOCATIONS[args.dataset], str(args.split_num)])
    df_train = pd.read_csv('/'.join([folder_path, 'train.csv']))
    df_test = pd.read_csv('/'.join([folder_path, 'test.csv']))

    df_train = df_train[[ROW_NAME_BY_DATASET[args.dataset], COL_NAME_BY_DATASET[args.dataset], PRICE_COL_NAME_BY_DATASET[args.dataset]]]
    df_test = df_test[[ROW_NAME_BY_DATASET[args.dataset], COL_NAME_BY_DATASET[args.dataset], PRICE_COL_NAME_BY_DATASET[args.dataset]]]

    df_train['label'] = 1.0
    df_test['label'] = 1.0

    df_train.columns = ['user', 'item', 'price', 'label']
    df_test.columns = ['user', 'item', 'price', 'label']

    train_data, data_info = DatasetFeat.build_trainset(df_train, None, None, None, None)
    data_dict = { 'n_users': data_info.n_users, 'n_items': data_info.n_items }
    del df_train
    
    test_data = DatasetFeat.build_testset(df_test)
    del df_test

    train_data.build_negative_samples(data_info, num_neg=1)
    test_data.build_negative_samples(data_info, num_neg=1)

    # Analyze which users and items actually exist in the train and test
    # datasets - only counting where an interaction exists
    # 
    train_users = np.unique(sp.find(train > 0)[0])
    test_users = np.unique(sp.find(test > 0)[0])
    train_items = np.unique(sp.find(train > 0)[1])
    test_items = np.unique(sp.find(test > 0)[1])

    cold_start_user_ratio = np.setdiff1d(test_users, np.intersect1d(train_users, test_users, assume_unique=True), assume_unique=True).shape[0] / test_users.shape[0]
    cold_start_item_ratio = np.setdiff1d(test_items, np.intersect1d(train_items, test_items, assume_unique=True), assume_unique=True).shape[0] / test_items.shape[0]


    # Each epoch is trained in a for loop, not through the library itself
    model_params = {
        'task': 'ranking',
        'embed_size': args.embed_size,
        'n_epochs': 1,
        'lr': args.learning_rate,
        'reg': None if args.reg is None else args.reg,
        'batch_size': args.batch_size,
        'hidden_units': "200,200,200",
        'data_info': data_info
    }

    # tf.compat.v1.reset_default_graph()
    model = DeepFM(**model_params)

    wandb_config = dict(**model_params, **data_dict)
    wandb_config['num_epochs'] = args.num_epochs
    wandb_config['dataset'] = args.dataset
    wandb_config['split_num'] = args.split_num
    wandb_config['cold_start_user_ratio'] = cold_start_user_ratio
    wandb_config['cold_start_item_ratio'] = cold_start_item_ratio

    wandb_run_name = str(datetime.now().strftime('%Y-%m-%d__%H%M%S')) + '_' + str(args.dataset) + '_' + str(args.split_num)
    wandb_group_name = 'DeepFM_' + args.dataset + '_CV'

    if not DEBUG:
        wandb.init(
            project="XXX",
            entity="XXX",
            group=wandb_group_name,
            name=wandb_run_name,
            config=wandb_config
        )

    prev_epoch = 0
    top_k = np.array([1, 2, 3, 4, 5, 10, 15])
    
    metrics_dict = dict()
    metrics_dict['epochs'] = []
    metrics_dict['train'] = dict()
    metrics_dict['test'] = dict()

    for k in top_k:
        label = 'prec_at_' + str(k)
        metrics_dict['train'][label] = []
        metrics_dict['test'][label] = []

    train = train.astype('int8')
    test = test.astype('int8')
    prices = prices.astype('int32')

    # Calculate average price per product on the sparse price data
    # 
    # "Count Nonzero" on sparse matrices
    # https://stackoverflow.com/questions/3797158/counting-non-zero-elements-within-each-row-and-within-each-column-of-a-2d-numpy
    # 
    num_interact_p_product = np.ma.array(np.diff(prices.tocsc().indptr))
    sum_prices_p_product = np.ma.array(np.array(np.sum(prices, axis=0), dtype='int32').reshape(-1))
    average_price_p_product = np.array(np.round(sum_prices_p_product / num_interact_p_product, 2))
    del num_interact_p_product
    del sum_prices_p_product
    del prices

    num_eval_chunks = np.ceil(np.multiply(test_users.shape[0], test.shape[1]) / NUMPY_ARRAY_CHUNKING_LIMIT).astype('int')
    row_chunks = np.array_split(test_users, num_eval_chunks)

    for epoch in range(1, args.num_epochs + 1):
        # model.fit(train_data, verbose=2, shuffle=True, metrics=["loss"])

        # lehl@2021-12-07: Implementation taken from libreco library, exposed here to
        # gather the train- and test loss values.
        # 
        if not model.graph_built:
            model._build_model()
            model._build_train_ops()
        model._check_has_sampled(train_data, 1)

        dg = DataGenFeat(train_data, False, False)
        train_losses = []
        for u, i, label, si, dv in dg(True, args.batch_size):
            f_d = model._get_feed_dict(u, i, si, dv, label, True)
            train_loss, _ = model.sess.run([model.loss, model.training_op], f_d)
            train_losses.append(train_loss)
        model.assign_oov()

        dg = DataGenFeat(test_data, False, False)
        test_losses = []
        for u, i, label, si, dv in dg(False, 4096):
            f_d = model._get_feed_dict(u, i, si, dv, label, False)
            test_loss, _ = model.sess.run([model.loss, model.training_op], f_d)
            test_losses.append(test_loss)

        wandb.log({'train_loss': np.sum(train_losses), 'test_loss': np.sum(test_losses)}, commit=False)

        precision = np.array([0 for k in top_k])
        recall = np.array([0 for k in top_k])
        ndcg = np.array([0 for k in top_k])
        revenue = np.array([0 for k in top_k])
        num_test_users = test_users.shape[0]
        
        for row_chunk in tqdm(row_chunks, desc=f"[Epoch {epoch}/{args.num_epochs}] Evaluating the {num_eval_chunks} chunks of the dataset."):
            num_users_in_chunk = row_chunk.shape[0]

            X_train = np.array(train.tocsr()[row_chunk,:].todense())
            X_test = np.array(test.tocsr()[row_chunk,:].todense())
            X_price = np.repeat([average_price_p_product], X_test.shape[0], axis=0)

            # Fix prediction to map back to the full user-item matrix space
            # ==================================================================
            # 
            train_users_in_chunk, indices_in_chunk, _ = np.intersect1d(train_users, row_chunk, assume_unique=True, return_indices=True)
            cold_start_users = np.setdiff1d(np.intersect1d(row_chunk, test_users, assume_unique=True), train_users_in_chunk, assume_unique=True)

            regular_prediction = predict(model, indices_in_chunk)

            # Generate Cold Start using AVERAGE Prediction in accordance to LibReco:
            # https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/ncf.py#L189
            # 
            cold_start_prediction = predict(model, np.repeat(data_info.n_users, cold_start_users.shape[0]))

            # Check which indices in the row_chunk need to be replaced with the regular
            # prediction or the cold start prediction
            # 
            row_chunk_indices = np.arange(row_chunk.shape[0])
            _, regular_users_idx, _ = np.intersect1d(row_chunk, train_users_in_chunk, assume_unique=True, return_indices=True)
            _, cold_start_users_idx, _ = np.intersect1d(row_chunk, cold_start_users, assume_unique=True, return_indices=True)

            all_item_indices = np.arange(train.shape[1])
            _, predicted_item_indices, _ = np.intersect1d(all_item_indices, train_items, assume_unique=True, return_indices=True)

            # Build the final prediction, taking into account which users and items
            # were in the training dataset and which ones are cold start users
            # 
            prediction = np.zeros((row_chunk.shape[0], train.shape[1]))
            prediction[regular_users_idx[:,None], predicted_item_indices[None,:]] = regular_prediction
            prediction[cold_start_users_idx[:,None], predicted_item_indices[None,:]] = cold_start_prediction

            # ======================================================================================
            # Regular Evaluation as with other methods from this point onwards
            # 
            # 
            # this is to not to recommend the products that the user already has.
            prediction[np.where(X_train==1)[0], np.where(X_train==1)[1]] = 0

            top_sorted_idx = np.argsort(-prediction)[:, :15]
            top_prices = X_price[np.arange(np.shape(X_price)[0])[:, np.newaxis], top_sorted_idx]

            eval_prec, eval_rec, eval_ndcg, eval_rev = user_precision_recall_ndcg__vectorized(top_sorted_idx,
                                                    X_test,
                                                    item_prices=top_prices,
                                                    at_ns=top_k)
            
            # Multiplied by the number of users in the eval chunk to 
            # match how much this chunk contributes to the total
            # 
            for k_id, k in enumerate(top_k):
                precision[k_id] += eval_prec[k_id] * num_users_in_chunk
                recall[k_id] += eval_rec[k_id] * num_users_in_chunk
                ndcg[k_id] += eval_ndcg[k_id] * num_users_in_chunk
                revenue[k_id] += eval_rev[k_id]

        # Divided by the total number of users to get a weighted average,
        # weighted by the number of users per evaluation chunk
        # 
        precision = np.divide(precision, num_test_users)
        recall = np.divide(recall, num_test_users)
        ndcg = np.divide(ndcg, num_test_users)

        f1_score = 2 * (precision*recall) / (precision+recall)

        wandb_dict = dict()
        wandb_dict['current_epoch'] = epoch
        wandb_dict['dataset'] = args.dataset
        wandb_dict['split_num'] = args.split_num
        for k_id, k in enumerate(top_k):
            wandb_dict['precision_at_' + str(k)] = precision[k_id]
            wandb_dict['recall_at_' + str(k)] = recall[k_id]
            wandb_dict['ndcg_at_' + str(k)] = ndcg[k_id]
            wandb_dict['revenue_at_' + str(k)] = revenue[k_id]
            wandb_dict['f1_at_' + str(k)] = f1_score[k_id]

        wandb_dict['cold_start_user_ratio'] = cold_start_user_ratio
        wandb_dict['cold_start_item_ratio'] = cold_start_item_ratio
        
        if not DEBUG:
            wandb.log(wandb_dict, commit=True)

            csv_file_path = 'Code_DeepFM/logs/' + wandb_run_name + '.csv'

            is_new_file = not os.path.exists(csv_file_path)
            open_mode = 'w' if is_new_file else 'a'

            with open(csv_file_path, open_mode) as f:
                writer = csv.DictWriter(f, wandb_dict.keys())

                if is_new_file:
                    writer.writeheader()
                writer.writerow(wandb_dict)

        prev_epoch = epoch

def user_precision_recall_ndcg__vectorized(predicted_indices, true_indices, item_prices=None,
                                           at_ns=np.array([1, 2, 3, 4, 5, 10, 15])):
    precisions = np.zeros(at_ns.shape[0])
    recalls = np.zeros(at_ns.shape[0])
    ndcgs = np.zeros(at_ns.shape[0])
    revenues = np.zeros(at_ns.shape[0])

    num_true_items_per_user = np.sum(true_indices, axis=1)
    num_true_items = np.sum(true_indices)

    for n_index, at_n in enumerate(at_ns):
        top_n_predicted_indices = predicted_indices[:, :at_n]
        # the ones that are in both prediction and ground truth.
        top_n_matching_indices = true_indices[np.arange(true_indices.shape[0])[
            :, None], top_n_predicted_indices]
        # number of correct recommendations per user.
        top_n_correct_counts = np.sum(top_n_matching_indices, axis=1)

        # (sum(number of matching items / index) / number of users)
        precisions[n_index] = np.divide(np.sum(np.divide(top_n_correct_counts.astype(float), at_n)),
                                        float(predicted_indices.shape[0]))

        # (sum(number of matching items / number of correct per user) / number of users)
        recalls[n_index] = np.divide(
            np.sum(np.divide(np.sum(top_n_matching_indices, axis=1),
                             num_true_items_per_user.astype(float))),
            float(predicted_indices.shape[0]))

        revenues[n_index] = np.sum(item_prices[:, :at_n] * top_n_matching_indices)

        dcg_lists = top_n_matching_indices
        dcg_values = np.apply_along_axis(lambda x: [(v / log(i + 1 + 1, 2)) for i, v in enumerate(x[:at_n])], 1,
                                         dcg_lists)
        dcg = np.sum(dcg_values, axis=1)

        ndcg_cal = 0

        for u in range(predicted_indices.shape[0]):
            # todo can we do it without loop?
            idcg_lists = [1 for i in range(num_true_items_per_user[u])]
            if len(idcg_lists) < at_n:
                idcg_lists += [0 for i in range(at_n - len(idcg_lists))]
            idcg_value = [(v / log(i + 1 + 1, 2))
                          for i, v in enumerate(idcg_lists[:at_n])]
            idcg = np.sum(idcg_value)
            ndcg_cal += dcg[u] / idcg
        ndcgs[n_index] = np.divide(ndcg_cal, float(predicted_indices.shape[0]))

    return precisions, recalls, ndcgs, revenues

def to_sparse(df, row_name, col_name, shape, data=None):
    # lehl@2021-08-26: To generate sparse matrices, it is simpler to 
    # use indices starting at 0 instead of 1 (where the data ids start)
    #
    # An additional value at the maximum boundary is added as point
    # (shape[0], shape[1]) with value 0, such that all datasets (train,
    # test, valid) match the same dimensionality
    #
    rows = np.append(df[row_name].to_numpy(), shape[0]) - 1
    cols = np.append(df[col_name].to_numpy(), shape[1]) - 1
    
    if data is not None:
        data = np.append(data, 0)
    else:
        data = np.ones(rows.shape[0])
        data[-1] = 0

    return csr_matrix((data, (rows, cols)), shape=shape)

def load_data_split(args):
    row_name = ROW_NAME_BY_DATASET[args.dataset]
    col_name = COL_NAME_BY_DATASET[args.dataset]
    price_col_name = PRICE_COL_NAME_BY_DATASET[args.dataset]

    # Default dataset_path is "none"-like and removed by the filter
    # 
    path_comps = [str(args.dataset_path), DATASET_LOCATIONS[args.dataset], str(args.split_num)]
    folder_path = '/'.join(list(filter(None, path_comps)))

    if not os.path.exists(folder_path):
        raise FileNotFoundError()

    train_path = '/'.join([folder_path, 'train.npz'])
    test_path = '/'.join([folder_path, 'test.npz'])
    price_path = '/'.join([folder_path, 'prices.npz'])
    
    start_time = time.time()

    if os.path.exists(train_path) and os.path.exists(test_path):
        print('Dataset already exists at ' + train_path + '. Loading from file...')
        train = load_npz(train_path)
        test = load_npz(test_path)
        prices = load_npz(price_path)

    else:
        print('Dataset is not processed, generating sparse matrices...')

        df_train = pd.read_csv('/'.join([folder_path, 'train.csv']))
        df_test = pd.read_csv('/'.join([folder_path, 'test.csv']))

        num_rows = np.max([ np.max(df_train[row_name]), np.max(df_test[row_name]) ])
        num_cols = np.max([ np.max(df_train[col_name]), np.max(df_test[col_name]) ])

        train = to_sparse(df_train, row_name=row_name, col_name=col_name, shape=(num_rows, num_cols))
        test = to_sparse(df_test, row_name=row_name, col_name=col_name, shape=(num_rows, num_cols))
        prices = to_sparse(df_test, row_name=row_name, col_name=col_name, shape=(num_rows, num_cols), data=df_test[price_col_name])

        save_npz(train_path, train)
        save_npz(test_path, test)
        save_npz(price_path, prices)

    print('Done after ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    return train, test, prices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='insurance')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--dataset_path', type=str, default='')

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--reg', type=float, default=0)
    args = parser.parse_args()

    train, test, prices = load_data_split(args)
    run_dfm(train, test, prices, args)

if __name__ == '__main__':
    main()
