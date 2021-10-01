import code
import sys
from collections import OrderedDict
from decouple import config, Csv
from wandb import wandb_run
import dataset_generation as ds_gen
import evaluation
import ml_models_general_plot as ml
import pandas as pd
import numpy as np
import file_paths
import pickle
from operator import itemgetter
import wandb
import cx_Oracle
from decouple import config
from math import log
from datetime import datetime

hyperparameter_defaults = dict(
    hidden_layer_sizes=100,
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.0001,
    early_stopping=False,
    # validation_fraction=0.1,
    # max_iter=100,
)

# ============================================================================
#   INPUT PARAMETERS, PARAMETER READOUT & DATABASE CREDENTIALS
# ============================================================================

# Input Parameters / Script Settings:
# =============================================
# - "product_id_colname":                    Name of the column containing the product_id
# - "customer_id_colname":                   Name of the column containing the user_id
# - "timestamp_colname":                     Name of the timestamp attribute used for temporal ordering
# - "revenue_colname":                       Name of the revenue attribute column (e.g. premium)
# - "dataset_order":                         In what order the dataset should be prepared, can be any of:
#                                               ['random', 'temporal']
# - "dataset_order_temporal_num_days":       Number of days that are used for temporal ordering (NYI)
# - "dataset_folder_ident":                  Name of the subfolder in the 'ds' folder, where intermediate
#                                               results are stored
# - "dataset_number":                        Number of the dataset
# - "dataset_type":                          Type of dataset, concerns the dataset generation (removal
#                                               probability or fixed target count), can be any of:
#                                               ['separate_product_cnt', 'mixed_product_cnt']
# - "test_split_percent":                    Split percentage for dividing training and test dataset,
#                                               passed to sklearn train_test_split method
# - "train_unordered":                       If the training should occur with unordered data
#                                               (TODO: confirm)
# - "target_cnt_product":                    How many products should be in the target. Only used with
#                                               :dataset_type equal to 'separate_product_cnt'
# - "removal_prob":                          What the probability is, that a product ends up in the target
#                                               matrix. Only used with :dataset_type equal to 'mixed_product_cnt'
# - "categorical_columns":                   Which additional customer features are used (apart from
#                                               which product were bought)
# - "stored_data_type":                      Defaults to 'pickle', otherwise intermediate results are
#                                               not saved
# - "printMatrices":                         Whether some debug prints are shown for the matrices used
# - "algorithm_key":                         Which algorithm should be used (see ml_models_general.py
#                                               for available models)
# - "normalize_probabilities":               If the probabilities should be normalized
# - "do_reranking":                          If Re-Ranking according to revenue should be used for evaluation
# - "do_top_n":                              If TopN should be used for evaluation
# - "top_n_values":                          Which values of N should be used for TopN (and Re-Ranking),
#                                               e.g. [1, 3, 5]
# - "cf_neighbor_ratio":                     What ratio of the total dataset should be considered as
#                                               "neighbors" for collaborative filtering
# - "load_inputdata_from":                   Whether to query the dataset again or used the stored files
#                                               "query" or "file" are the possible options
USE_WANDB = False
if len(sys.argv) > 1 and not USE_WANDB:
    algorithm_key = sys.argv[1]
    removal_prob = float(sys.argv[2])
    ordering = sys.argv[3]
    dataset_type = sys.argv[4]
    target_cnt_product = int(sys.argv[5])
    NUM_SPLIT = int(sys.argv[6])
else:
    # ['Baseline', 'RandomForest', 'LogisticRegression', 'CollaborativeFiltering', 'NN', 'SPM']
    algorithm_key = 'NN'
    removal_prob = 0.7
    ordering = 'temporal'
    dataset_type = 'separate_product_cnt'
    target_cnt_product = 5
    NUM_SPLIT = 5
print("sys.argv", sys.argv)

# WANDB Initialization
#
WANDB_API_KEY = ''
wandb_group_name = 'MLP_Scikit_' + str(NUM_SPLIT)
wandb_run_name = str(datetime.now().strftime('%Y-%m-%d__%H%M%S'))

wandb.init(entity="yasies93", project="zhaw_nquest", group=wandb_group_name,
           name=wandb_run_name, config=hyperparameter_defaults)

wandb.log({'current_epoch': 1})

has_price = True
NUM_CUSTOMERS = 
dataset_name = ''
rm_user_features = True
input_params = {'product_id_colname': 'customer_id',
                'customer_id_colname': 'product_id',
                'timestamp_colname': 'date',
                'revenue_colname': 'Price',
                'dataset_order': ordering,
                'dataset_order_temporal_num_days': 0,
                'dataset_folder_ident': '',  # + algorithm_key,
                'dataset_number': str(NUM_SPLIT) + '_split',
                'dataset_type': dataset_type,
                'test_split_percent': 0.3,
                'train_unordered': True,
                'target_cnt_product': target_cnt_product,
                'removal_prob': removal_prob,
                'categorical_columns': [],
                'stored_data_type': 'pickle',
                'printMatrices': True,
                'algorithm_key': algorithm_key,
                'normalize_probabilities': False,
                'do_reranking': False,
                'do_top_n': True,
                'top_n_values': [1, 2, 3, 4, 5, 10, 15],
                'cf_neighbor_ratio': 0.01,
                'load_inputdata_from': 'file'}

# ML_Algorithm = Algorithm
csv_data = OrderedDict()
csv_data['Algorithm'] = input_params['algorithm_key']
csv_data['Dataset'] = input_params['dataset_folder_ident']
csv_data['DatasetNumber'] = input_params['dataset_number']
csv_data['DatasetType'] = input_params['dataset_type']
csv_data['RemovalProbability'] = input_params['removal_prob']
csv_data['TargetCount'] = input_params['target_cnt_product']
csv_data['Ordering'] = input_params['dataset_order']
csv_data['ds_creation_time'] = '-'
csv_data['ds_preprocessing_time'] = '-'
csv_data['ds_training_time'] = '-'
csv_data['ds_prediction_time'] = '-'

# ------------------------------------------------------------------------------------
#                                   Functions Start
# ------------------------------------------------------------------------------------
# evaluation
# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
# calculate NDCG@k


def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2))
                 for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2))
                  for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


def user_precision_recall_ndcg(new_user_prediction, test, products_prices):
    dcg_list = []
    # compute the number of true positive items at top k
    count_1, count_2, count_3, count_4, count_5, count_10, count_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    revenue_1, revenue_2, revenue_3, revenue_4, revenue_5, revenue_10, revenue_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(15):
        prediction_condition = new_user_prediction[i][0] in test
        if i == 0 and prediction_condition:
            count_1 = 1.0
            revenue_1 = products_prices[i]
        if i < 2 and prediction_condition:
            count_2 += 1.0
            revenue_2 += products_prices[i]
        if i < 3 and prediction_condition:
            count_3 += 1.0
            revenue_3 += products_prices[i]
        if i < 4 and prediction_condition:
            count_4 += 1.0
            revenue_4 += products_prices[i]
        if i < 5 and prediction_condition:
            count_5 += 1.0
            revenue_5 += products_prices[i]
        if i < 10 and prediction_condition:
            count_10 += 1.0
            revenue_10 += products_prices[i]
        if prediction_condition:
            count_15 += 1.0
            revenue_15 += products_prices[i]
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, 1)
    ndcg_tmp_2 = NDCG_at_k(dcg_list, idcg_list, 2)
    ndcg_tmp_3 = NDCG_at_k(dcg_list, idcg_list, 3)
    ndcg_tmp_4 = NDCG_at_k(dcg_list, idcg_list, 4)
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, 5)
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, 10)
    ndcg_tmp_15 = NDCG_at_k(dcg_list, idcg_list, 15)

    # precision@k
    precision_1 = count_1
    precision_2 = count_2 / 2.0
    precision_3 = count_3 / 3.0
    precision_4 = count_4 / 4.0
    precision_5 = count_5 / 5.0
    precision_10 = count_10 / 10.0
    precision_15 = count_15 / 15.0
    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_2 = count_2 / l
    recall_3 = count_3 / l
    recall_4 = count_4 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_15 = count_15 / l
    # return precision, recall
    return np.array([precision_1, precision_2, precision_3, precision_4, precision_5, precision_10, precision_15]),\
        np.array([recall_1, recall_2, recall_3, recall_4, recall_5, recall_10, recall_15]), \
        np.array([ndcg_tmp_1, ndcg_tmp_2, ndcg_tmp_3, ndcg_tmp_4, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15]), \
        np.array([revenue_1, revenue_2, revenue_3, revenue_4,
                  revenue_5, revenue_10, revenue_15])


def evaluate_jcastyle(prediction, test_mask, train_mask, price_mask):
    precision_1, precision_2, precision_3, precision_4, precision_5, \
        precision_10, precision_15 = \
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_2, recall_3, recall_4, recall_5, \
        recall_10, recall_15 = \
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_2, ndcg_3, ndcg_4, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    revenue_1, revenue_2, revenue_3, revenue_4, revenue_5, \
        revenue_10, revenue_15 = \
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
    precision = \
        np.array([precision_1, precision_2, precision_3, precision_4,
                  precision_5, precision_10, precision_15])
    recall = \
        np.array([recall_1, recall_2, recall_3,
                  recall_4, recall_5, recall_10, recall_15])
    ndcg = \
        np.array([ndcg_1, ndcg_2, ndcg_3,
                  ndcg_4, ndcg_5, ndcg_10, ndcg_15])
    revenue = \
        np.array([revenue_1, revenue_2, revenue_3,
                  revenue_4, revenue_5, revenue_10, revenue_15])

    prediction = prediction + train_mask * -100000.0

    masked = np.ma.masked_equal(price_mask, 0)
    price_mask = np.repeat([masked.mean(axis=0)], price_mask.shape[0], axis=0)
    user_num = prediction.shape[0]
    prod_train_test_cnt = [int(cnt) if cnt < 5 else 5 for cnt in (
        train_mask.sum(axis=1)+test_mask.sum(axis=1))]
    metrics_by_product_count = {'revenue': {}, 'precision': {}, 'user_cnt': {}}
    Insuranceall_results_df = pd.DataFrame(
        columns=['user', 'top15_predictions', 'product_in_GT', 'top15_price'])
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        # the indices of the true positive items in the test set
        u_test = np.where(u_test == 1)[0]
        u_pred = prediction[u, :]
        u_price = price_mask[u, :]
        # sum user products in train per user.
        u_prod_train_test_cnt = prod_train_test_cnt[u]
        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train,
                           u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)
        indices = [int(i[0]) for i in top15]
        top15_price = u_price[indices]
        # calculate the metrics
        if not len(u_test) == 0:
            Insuranceall_results_df.loc[u] = [u, top15, u_test, top15_price]
            precision_u, recall_u, ndcg_u, revenue_u = user_precision_recall_ndcg(
                top15, u_test, top15_price)
            try:
                metrics_by_product_count['revenue'][f'{u_prod_train_test_cnt}_product'] += revenue_u
                metrics_by_product_count['user_cnt'][f'{u_prod_train_test_cnt}_product'] += 1
            except KeyError:
                metrics_by_product_count['revenue'][f'{u_prod_train_test_cnt}_product'] = revenue_u
                metrics_by_product_count['user_cnt'][f'{u_prod_train_test_cnt}_product'] = 1

            try:
                metrics_by_product_count['precision'][f'{u_prod_train_test_cnt}_product'] += precision_u
            except KeyError:
                metrics_by_product_count['precision'][f'{u_prod_train_test_cnt}_product'] = precision_u

            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
            revenue += revenue_u
        else:
            user_num -= 1
    print("metrics for all users calculated.")
    Insuranceall_results_df.to_csv('./Insurance_recom_vectors.csv', index=False)
    metrics_by_product_count_df = pd.DataFrame.from_dict(
        data=metrics_by_product_count, orient='index')
    for n_products in metrics_by_product_count_df:
        metrics_by_product_count_df.at['precision',
                                       n_products] /= metrics_by_product_count_df.at['user_cnt', n_products]

    # compute the average over all users
    precision /= user_num
    recall /= user_num
    ndcg /= user_num
    print('precision_1\t[%.7f],\t||\t precision_2\t[%.7f],\t||\t precision_3\t[%.7f],\t||\t precision_4\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]'
          % (precision[0],
             precision[1],
             precision[2],
             precision[3],
             precision[4],
             precision[5],
             precision[6],))
    print('recall_1   \t[%.7f],recall_2   \t[%.7f],recall_3   \t[%.7f],recall_4   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]'
          % (recall[0],
             recall[1],
             recall[2],
             recall[3],
             recall[4],
             recall[5],
             recall[6]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] +
                                                    recall[0]) if not precision[0] + recall[0] == 0 else 0
    f_measure_2 = 2 * (precision[1] * recall[1]) / (precision[1] +
                                                    recall[1]) if not precision[1] + recall[1] == 0 else 0
    f_measure_3 = 2 * (precision[2] * recall[2]) / (precision[2] +
                                                    recall[2]) if not precision[2] + recall[2] == 0 else 0
    f_measure_4 = 2 * (precision[3] * recall[3]) / (precision[3] +
                                                    recall[3]) if not precision[3] + recall[3] == 0 else 0
    f_measure_5 = 2 * (precision[4] * recall[4]) / (precision[4] +
                                                    recall[4]) if not precision[4] + recall[4] == 0 else 0
    f_measure_10 = 2 * (precision[5] * recall[5]) / (precision[5] +
                                                     recall[5]) if not precision[5] + recall[5] == 0 else 0
    f_measure_15 = 2 * (precision[6] * recall[6]) / (precision[6] +
                                                     recall[6]) if not precision[6] + recall[6] == 0 else 0
    print('f_measure_1\t[%.7f],\t||\t f_measure_2\t[%.7f],\t||\t f_measure_3\t[%.7f],\t||\t f_measure_4\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]'
          % (f_measure_1,
             f_measure_2,
             f_measure_3,
             f_measure_4,
             f_measure_5,
             f_measure_10,
             f_measure_15))
    f_score = [f_measure_1, f_measure_2, f_measure_3,
               f_measure_4, f_measure_5, f_measure_10, f_measure_15]
    print('ndcg_1     \t[%.7f],ndcg_2     \t[%.7f],ndcg_3     \t[%.7f],ndcg_4     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]'
          % (ndcg[0],
             ndcg[1],
             ndcg[2],
             ndcg[3],
             ndcg[4],
             ndcg[5],
             ndcg[6]))
    print('revenue_1  \t[%.7f],\t||\t revenue_2  \t[%.7f], \t||\t revenue_3  \t[%.7f], \t||\t revenue_4  \t[%.7f], \t||\t revenue_5  \t[%.7f],\t||\t revenue_10  \t[%.7f],\t||\t revenue_15  \t[%.7f]'
          % (revenue[0],
              revenue[1],
              revenue[2],
              revenue[3],
              revenue[4],
              revenue[5],
              revenue[6]))
    print('metrics_by_product_count_df["revenue"]:',
          metrics_by_product_count_df.loc['revenue', :])
    print('metrics_by_product_count_df["precision"]:',
          metrics_by_product_count_df.loc['precision', :])
    # store the results in csv.
    metrics = {'precision': precision, 'recall': recall,
               'f_score': f_score, 'revenue': revenue, 'ndcg': ndcg}
    metrics_by_product_count_df.to_csv(
        f'../Final_results_loss/Insurance/{algorithm_key}_results_userswithNproducts_{NUM_SPLIT}.csv')
    results_metrics = pd.DataFrame.from_dict(
        data=metrics, orient='index').T[['precision', 'recall', 'f_score', 'revenue', 'ndcg']].to_numpy().flatten()

    np.savetxt(f'../Final_results_loss/Insurance/{algorithm_key}_results_metrics_{NUM_SPLIT}.csv',
               results_metrics.reshape(1, results_metrics.shape[0]),
               delimiter=",")

    return precision, recall, f_score, revenue, ndcg

# ------------------------------------------------------------------------------------
#                                   Functions End
# ------------------------------------------------------------------------------------
# ============================================================================
#   LOAD AND GENERATE DATASET
# ============================================================================


# --> The following lines can be changed for a new dataset, providing the necessary
# base data to generate the dataset initially
#
if input_params['load_inputdata_from'] == 'query':
    DB = config('INSURANCE_DB')
    USERNAME = config('INSURANCE_USER')
    PASS = config('INSURANCE_PASSWORD')
    DB_URL = config('INSURANCE_URL')
    PORT = config('INSURANCE_PORT', cast=int)
    SID = config('INSURANCE_SID')
    DATASET_TYPE = 'oracle'

    CUSTOMER_FEATURES_PATH = 'ds/' + \
        input_params['dataset_folder_ident'] + '/customer_features.csv'
    CUSTOMER_PRODUCTS_PATH = 'ds/' + \
        input_params['dataset_folder_ident'] + '/customer_products.csv'

    def customer_products_query(num_rows=100):
        base_query = """
            INSERT PRODUCT QUERY
            """

        if num_rows > 0:
            return base_query + ' FETCH NEXT ' + str(num_rows) + ' ROWS ONLY'
        else:
            return base_query

    def customer_features_query():
        return """ INSERT FEATURE QUERY"""

    def query_row_count(cursor, query):
        cursor.execute("SELECT COUNT(*) FROM ({})".format(query))
        return cursor.fetchone()[0]

    def query_result_from_slices(query, csv_path, slice_size=2**15):
        try:
            df = pd.read_csv(csv_path)
        except (OSError, FileNotFoundError):
            dsn_tns = cx_Oracle.makedsn(DB_URL, PORT, SID)
            conn = cx_Oracle.connect(USERNAME, PASS, dsn_tns)
            print("connected to oracle. ", conn)
            cursor = conn.cursor()

            num_rows = query_row_count(cursor, query)
            num_slices = num_rows // slice_size + 1
            count = 0
            result_list = []

            cursor.execute(query)

            # Evaluate column names
            #
            columns = []
            for col in cursor.description:
                columns.append(col[0])

            # Fetch rows in slices
            #
            for i in range(num_slices):
                print("Fetching slices {}/{}...".format(i+1, num_slices), end='')
                result = cursor.fetchmany(numRows=slice_size)
                print('done')

                num_in_slice = len(result)
                count += num_in_slice
                print("\t{} new, {}/{}".format(num_in_slice, count, num_rows))

                result_list.append(pd.DataFrame(result, columns=columns))

            df = pd.concat(result_list)
            df.to_csv(csv_path)

        return df

    # Load :customer_features
    # --------------------------------------------
    # Shape:    (num_customers, num_features)
    #
    # Pandas Dataframe, where the columns are customer features that do
    # NOT include product buying statistics, for example age or gender.
    #
    # Index is not a particular attribute, but the customer_id column should be part
    # of this matrix, e.g. 'User_ID' (see input_params['customer_id_colname'])
    #
    customer_features = query_result_from_slices(
        customer_features_query(), CUSTOMER_FEATURES_PATH)
    customer_features = customer_features[[
        'customer_id'] + input_params['categorical_columns']]

    customer_features.gender.fillna(-1, inplace=True)
    customer_features.industry_code.fillna(-1, inplace=True)

    # Load :customer_products
    # --------------------------------------------
    # Shape:    (num_transactions, 2-4)
    #
    # Pandas Dataframe, where the columns represent transaction-style
    # entries containing the CustomerID, the ProductID as well as optional
    # Timestamp and Price/Revenue/Premium information:
    #
    # [CustomerID, ProductID, (Timestamp), (Price)]
    #
    customer_products = query_result_from_slices(
        customer_products_query(-1), CUSTOMER_PRODUCTS_PATH)
    customer_products.date.fillna('9999-12-31', inplace=True)

    # Select only company or private customers, if chosen
    #
    if input_params['dataset_folder_ident'] == 'insurance_company':
        customer_features = customer_features[customer_features.segment == 2]
    elif input_params['dataset_folder_ident'] == 'insurance_private':
        customer_features = customer_features[customer_features.segment == 1]

    # Randomly pick NUM_CUSTOMERS customers from all customers
    #
    unique_customer_ids = np.intersect1d(
        customer_features.customer_id.unique(), customer_products.customer_id.unique())
    np.random.seed(42)
    chosen_customer_ids = np.random.choice(
        unique_customer_ids, NUM_CUSTOMERS, replace=False)

    # NUM_CUSTOMERS = 5
    # np.random.seed(42)
    # customer_ids_with_products = np.unique(np.array(customer_products.customer_id))
    # customer_ids_with_features = np.unique(np.array(customer_features.customer_id))
    # customer_ids_without_products = np.setdiff1d(customer_ids_with_features, customer_ids_with_products)
    # chosen_customer_ids = np.concatenate((
    #     customer_products.groupby('customer_id').count().sort_values('product').tail(NUM_CUSTOMERS).index.to_numpy(),
    #     np.random.choice(customer_ids_without_products, NUM_CUSTOMERS, replace=False)))

    customer_features = customer_features[customer_features['customer_id'].isin(
        chosen_customer_ids)]
    customer_products = customer_products[customer_products['customer_id'].isin(
        chosen_customer_ids)]

    # Use loaded files and Pandas DF to create the dataset (as it has not been generated before)
    # --> These few lines should not be changed when a new dataset is used
    # ============================================================================
    #
    dataset, csv_data['ds_creation_time'] = ds_gen.generate(
        customer_features, customer_products, input_params)
elif input_params['load_inputdata_from'] == 'file':
    # @esla from
    print("before storing the dataset_dict pickle files.")
    # fill in the dataset file:b
    dataset = {}
    dataset_full = {}

    if dataset_name == 'ml-1m_skewed':
        data_path = '../data/ml-1m_skewed/nomaxprod/' + \
            str(NUM_SPLIT)+'/'
    elif dataset_name == 'ml-1m':
        data_path = '../data/ml-1m/min6/10fold_CV/' + \
            str(NUM_SPLIT)+'/'
    elif dataset_name == 'insurance':
        data_path = '../data/insurance/' + \
            str(NUM_SPLIT)+'/'

    with open(data_path + 'data_splits_test_target.pkl', 'rb') as f:
        dataset['Y_test'] = pickle.load(f)
        print("y_test loaded ... ")
    with open(data_path + 'data_splits_test_feature.pkl', 'rb') as f:
        dataset['X_test'] = pickle.load(f)
    print("x_test loaded ... ")
    with open(data_path + 'data_splits_train_feature.pkl', 'rb') as f:
        dataset['X_train'] = pickle.load(f)
    print("x_train loaded ... ")
    with open(data_path + 'data_splits_train_target.pkl', 'rb') as f:
        dataset['Y_train'] = pickle.load(f)
    print("y_train loaded ... ")
    if has_price:
        with open(data_path + 'data_splits_train_price.pkl', 'rb') as f:
            dataset['price_train'] = pickle.load(f)
        print("price train loaded ... ")
        with open(data_path + 'data_splits_test_price.pkl', 'rb') as f:
            dataset['price_test'] = pickle.load(f)
    else:
        dataset['price_train'] = \
            pd.DataFrame(0,
                         index=dataset['Y_train'].index,
                         columns=dataset['Y_train'].columns)
        dataset['price_test'] = \
            pd.DataFrame(0,
                         index=dataset['Y_test'].index,
                         columns=dataset['Y_test'].columns)

    print("price test loaded ... ")
    if rm_user_features:
        dataset['X_test'] = dataset['X_test'].iloc[:,
                                                   :dataset['Y_test'].shape[1]]
        dataset['X_train'] = dataset['X_train'].iloc[:,
                                                     :dataset['Y_test'].shape[1]]

    # code.interact(local=dict(globals(), **locals()))
# remove the users with less than 5 products from the train data.
# but still use that for train mask during evaluation.
# dataset_full['X_train'] = dataset['X_train']
# dataset_full['Y_train'] = dataset['Y_train']
# dataset['X_train'] = dataset['X_train'][dataset['X_train'].sum(axis=1)>3]
# dataset['Y_train'] = dataset['Y_train'].loc[dataset['X_train'].index, :]
# @esla till
# ============================================================================
#   PREPROCESSING
# ============================================================================
print("ds preprocessing...")
dataset, csv_data['ds_preprocessing_time'] = ml.preprocessing(
    dataset, input_params)
# average the price over nonzero values
# avg train
dataset['price_train'] = np.ma.masked_equal(
    dataset['price_train'], 0).mean(axis=0).filled(0)
ones = np.ones(dataset['X_train'].shape)
dataset['price_train'] = np.multiply(
    ones, dataset['price_train'], dtype="float32")
# avg test
dataset['price_test'] = np.ma.masked_equal(
    dataset['price_test'], 0).mean(axis=0).filled(0)
ones = np.ones(dataset['X_test'].shape)
dataset['price_test'] = np.multiply(
    ones, dataset['price_test'], dtype="float32")
# code.interact(local=dict(globals(), **locals()))
# ============================================================================
#   TRAIN THE MODEL
# ============================================================================
print("training...")
model, transform, csv_data['ds_training_time'] = ml.train(
    dataset, input_params, hyperparameter_defaults)

# ============================================================================
#   EVALUATE THE MODEL
# ============================================================================
# csv_data = evaluation.evaluate(
#     dataset, model, transform, input_params, csv_data=csv_data)


# ============================================================================
#   EVALUATE JCA STYLE
# ============================================================================
print("evaluating...")
predictions_proba = model.predict_proba(dataset['X_test'])
print("prediction proba available")
test_mask = dataset['Y_test']
dataset['X_train'] = \
    dataset['X_train'][:, :dataset['Y_train'].shape[1]]
dataset['X_test'] = \
    dataset['X_test'][:, :dataset['Y_train'].shape[1]]
train_mask = dataset['X_train'] + dataset['X_test'] + dataset['Y_train']
price_mask = dataset['price_test']

precision, recall, f_score, revenue, ndcg = evaluate_jcastyle(
    predictions_proba, test_mask, train_mask, price_mask)


if USE_WANDB:
    precision_at1 = precision[0]
    recall_at1 = recall[0]
    f1_at1 = f_score[0]
    wandb.log({'precision_at1': precision_at1,
               'recall_at1': recall_at1,
               'f1_at1': f1_at1},
              step=0)
