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
import pandas as pd
import numpy as np

from scipy.stats import skew

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

def load_data_split(args, split_num):
    row_name = ROW_NAME_BY_DATASET[args.dataset]
    col_name = COL_NAME_BY_DATASET[args.dataset]
    price_col_name = PRICE_COL_NAME_BY_DATASET[args.dataset]

    folder_path = '/'.join([DATASET_LOCATIONS[args.dataset], str(split_num)])

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='insurance')
    args = parser.parse_args()

    train, test, _ = load_data_split(args, split_num=1)
    combined = train + test

    histogram = np.array(np.sum(combined,axis=0)).reshape(-1)
    skewness = skew(histogram)

    print(f"Pearson Coefficient of Skewness for Dataset {args.dataset}:\t{skewness}")

    print('')
    interactions_per_item = np.sum(combined, axis=0)
    print(f"Min Interactions p. Item {np.min(interactions_per_item)}")
    print(f"Mean Interactions p. Item {np.mean(interactions_per_item)}")
    print(f"Max Interactions p. Item {np.max(interactions_per_item)}")

    print('')
    interactions_per_user = np.sum(combined, axis=1)
    print(f"Min Interactions p. User {np.min(interactions_per_user)}")
    print(f"Mean Interactions p. User {np.mean(interactions_per_user)}")
    print(f"Max Interactions p. User {np.max(interactions_per_user)}")

    import code; code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()
