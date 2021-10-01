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
import numpy as np

import wandb
import implicit
from implicit.als import _als

# Max number of interactions per evaluation chunk
# (~= dense matrix dimensions)
NUMPY_ARRAY_CHUNKING_LIMIT = 16384 * 16384 # 2**28
DEBUG = False

DATASET_LOCATIONS = {
    'insurance': 'data/insurance',
    'ml1m_min6': 'data/ml-1m/min6',
    'ml1m_max5': 'data/ml-1m/max5oldest',
    'yoochoose': 'data/yoochoose',
    'yoochoose_small': 'data/yoochoose_small',
    'retailrocket': 'data/retailrocket'
}

ROW_NAME_BY_DATASET = {
    'insurance': 'customer_id',
    'ml1m_min6': 'customer_id',
    'ml1m_max5': 'customer_id',
    'yoochoose': 'Session ID',
    'yoochoose_small': 'Session ID',
    'retailrocket': 'visitorid'
}

COL_NAME_BY_DATASET = {
    'insurance': 'product',
    'ml1m_min6': 'product_id',
    'ml1m_max5': 'product_id',
    'yoochoose': 'Item ID',
    'yoochoose_small': 'Item ID',
    'retailrocket': 'itemid'
}

PRICE_COL_NAME_BY_DATASET = {
    'insurance': 'annual_premium',
    'ml1m_min6': 'Price',
    'ml1m_max5': 'Price',
    'yoochoose': 'Price',
    'yoochoose_small': 'Price',
    'retailrocket': 'Price'
}


class AlternatingLeastSquaresPredictable(implicit.als.AlternatingLeastSquares):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Simplification of the "rank_items" function that works on the full matrix, see for comparison:
    # https://github.com/benfred/implicit/blob/471194c8774ec4cd2c9fae294d113762fc505962/implicit/recommender_base.pyx#L301
    #
    # lehl@2021-09-07: In case that user_rows are provided
    #   (the indices), only use those for the prediction
    #
    def predict(self, user_rows=None):
        if user_rows is not None:
            item_factors = self.item_factors[user_rows, :]
            return np.inner(item_factors, self.user_factors)
            
        return np.inner(self.item_factors, self.user_factors)

    # Calculates the loss values for the given item_user matrix, which could be
    # train or test (and give the corresponding loss values)
    # 
    def calculate_loss(self, item_users):
        Ciu = item_users

        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            Ciu = Ciu.tocsr()

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)

        Cui = Ciu.T.tocsr()

        return _als.calculate_loss(Cui, self.user_factors, self.item_factors, self.regularization, num_threads=self.num_threads)


def run_als(train, test, prices, args):
    model_params = {
        'factors': args.factors,
        'regularization': args.reg,
        'use_gpu': False
    }
    model = AlternatingLeastSquaresPredictable(**model_params)

    wandb_config = model_params.copy()
    wandb_config['num_epochs'] = args.num_epochs

    wandb_run_name = str(datetime.now().strftime('%Y-%m-%d__%H%M%S')) + '_split' + str(args.split_num)
    wandb_group_name = 'ALS_CV_' + args.dataset
    
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

    # users that are in the test dataset but also have products
    test_users = np.unique(sp.find(test > 0)[0])

    num_eval_chunks = np.ceil(np.multiply(test_users.shape[0], test.shape[1]) / NUMPY_ARRAY_CHUNKING_LIMIT).astype('int')
    row_chunks = np.array_split(test_users, num_eval_chunks)

    for epoch in range(1, args.num_epochs + 1):
        model.iterations = epoch - prev_epoch
        model.fit(train)

        # Calculate the loss values for ALS
        train_loss = model.calculate_loss(train)
        test_loss = model.calculate_loss(test)

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

            prediction = model.predict(row_chunk)
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
        wandb_dict['train_loss'] = train_loss
        wandb_dict['test_loss'] = test_loss
        for k_id, k in enumerate(top_k):
            wandb_dict['precision_at_' + str(k)] = precision[k_id]
            wandb_dict['recall_at_' + str(k)] = recall[k_id]
            wandb_dict['ndcg_at_' + str(k)] = ndcg[k_id]
            wandb_dict['revenue_at_' + str(k)] = revenue[k_id]
            wandb_dict['f1_at_' + str(k)] = f1_score[k_id]
        
        if not DEBUG:
            wandb.log(wandb_dict, commit=True)

            csv_file_path = 'Code_ALS/logs/' + wandb_run_name + '_' + str(args.dataset) + '_cv' + str(args.split_num) + '.csv'
            
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

    folder_path = '/'.join([DATASET_LOCATIONS[args.dataset], str(args.split_num)])

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
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--reg', type=float, default=1e-2)
    parser.add_argument('--factors', type=int, default=100)

    args = parser.parse_args()

    train, test, prices = load_data_split(args)
    run_als(train, test, prices, args)

if __name__ == '__main__':
    # lehl@2021-08-26: Given as a warning by implicit:
    # WARNING:root:OpenBLAS detected. Its highly recommend to set
    # the environment variable 'export OPENBLAS_NUM_THREADS=1' to
    # disable its internal multithreading
    # 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    main()
