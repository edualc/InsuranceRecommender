from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import time
import datetime
import h5py
import os
from decouple import config
from collaborative_filtering import CollaborativeFiltering
# from diversity_metrics import metric__inverse_gini_coefficient, metric__mean_self_information, metric__aggregate_diversity
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sklearn.metrics as metrics
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import neural_network
import pickle
import file_paths
from scipy import sparse
from scipy.special import xlogy
import matplotlib.pyplot as plt
from operator import itemgetter
from math import log
import code
import wandb
from wandb_decorator import *

# import interface2
# Folder where data is stored with trained models
#
TRAINED_MODEL_FOLDER = './trained_models/'

# Wrapper method around the preprocessing__execution to deal
# with loading preprocessed data, if available
#
# inja
# Claude vectorized eval


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

        item_prices[:, :at_n] = item_prices[:, :at_n] * top_n_matching_indices
        revenues[n_index] = np.sum(item_prices[:, :at_n])

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


def test_model_all(prediction, test_mask, train_mask, price_mask):
    # this is to not to recommend the products that the user already has.
    prediction = prediction + train_mask * -100000.0
    # get average of the prices for each product among nonzero values
    masked = np.ma.masked_equal(price_mask, 0)
    price_mask = np.repeat([masked.mean(axis=0)], price_mask.shape[0], axis=0)

    a = datetime.datetime.now()
    print("Starting evaluation: ", a)

    # EVALUATION: NOW WITH NUMPY!
    # ===============================================================================
    # lehl@2021-04-28:  Since users without test products are skipped for
    #                   calculating the evaluation metrics, filter here
    #                   already to reduce the amount of users that have
    #                   to be processed
    #

    user_ids_with_test_products = np.unique(np.where(test_mask == 1)[0])
    print("\t\tNumber of users to be evaluated: {} (out of {} total)".format(user_ids_with_test_products.shape[0],
                                                                             prediction.shape[0]))

    # Apply the filter, such that only the data of relevant users is available
    #
    filtered_predictions = prediction[user_ids_with_test_products, :]
    filtered_price = price_mask[user_ids_with_test_products, :]
    filtered_true_labels = test_mask[user_ids_with_test_products, :]

    # Selecting the indices in the top 15
    #
    top15_sorted_idx = np.argsort(-filtered_predictions)[:, :15]
    top15_prices = filtered_price[np.arange(np.shape(filtered_price)[0])[
        :, np.newaxis], top15_sorted_idx]
    AT_NS = np.array([1, 2, 3, 4, 5, 10, 15])
    
    precision, recall, ndcg, revenue = user_precision_recall_ndcg__vectorized(top15_sorted_idx,
                                                                              filtered_true_labels,
                                                                              item_prices=top15_prices,
                                                                              at_ns=AT_NS)

    b = datetime.datetime.now()
    print("Evaluation took: ", b - a)

    def print_np_array(arr, label):
        print_list = list()

        for i, at_n in enumerate(AT_NS):
            print_list.append(str(label) + '_@_' + str(at_n) +
                              '\t' + "{:.7f}".format(arr[i]))

        print('\t||\t'.join(print_list))

    f_score = 2 * (precision * recall) / (precision + recall)
    f_score[np.isnan(f_score)] = 0

    print_np_array(precision, 'precision')
    print_np_array(recall, 'recall')
    print_np_array(f_score, 'f_measure')
    print_np_array(ndcg, 'ndcg')
    print_np_array(revenue, 'revenue')
    print("\n")

    # # store the recommendations
    # #
    # print("storing to file...")
    # # 'user' @ dataframe below

    # user_ids_to_save = user_ids_with_test_products

    # # 'top15_predictions_withscode' @ dataframe below
    # top15_predictions_to_save = filtered_predictions[
    #     np.arange(np.shape(filtered_predictions)[0])[:, np.newaxis], top15_sorted_idx]

    # # 'top_15_indices' @ dataframe below
    # top15_indices_to_save = top15_sorted_idx

    # # 'product_in_GT' @ dataframe below
    # ground_truth_products = np.array(
    #     [np.array([], dtype='int64') if np.nonzero(row)[0].size == 0 else np.nonzero(row)[0] for row in
    #      filtered_true_labels])

    # # 'u_current_prod' @ dataframe below
    # existing_products = np.array(
    #     [np.array([], dtype='int64') if np.nonzero(row)[0].size == 0 else np.nonzero(row)[0] for row in
    #      train_mask[user_ids_with_test_products, :]])

    # # 'u_price' @ dataframe below
    # top15_prices_to_save = top15_prices

    # # Create and save the prediction results
    # #
    # Insuranceall_results_df = pd.DataFrame(
    #     data=[user_ids_to_save, top15_predictions_to_save, top15_indices_to_save, ground_truth_products,
    #           existing_products, top15_prices_to_save]).T
    # Insuranceall_results_df.columns = ['user', 'top15_predictions_withscode', 'top_15_indices', 'product_in_GT',
    #                                 'u_current_prod', 'u_price']
    # Insuranceall_results_df.to_csv('./Insurance_fromdf_res2.csv', index=False)

    c = datetime.datetime.now()
    print("Collecting and saving the prediction data took: ", c - b)

    # # store the results in csv.
    # metrics = {'precision': precision, 'recall': recall, 'f_score': f_score, 'revenue': revenue}
    # pd.DataFrame.from_dict(data=metrics, orient='index').to_csv('results_metrics.csv')

    return precision, recall, f_score, revenue, ndcg

    # return precision, recall, f_score, ndcg, revenue, Insuranceall_results_df


def metric_record(precision, recall, f_score, NDCG, args,
                  metric_path):  # record all the results' details into files
    path = metric_path + '.txt'

    with open(path, 'w') as f:
        f.write(str(args) + '\n')
        f.write('precision:' + str(precision) + '\n')
        f.write('recall:' + str(recall) + '\n')
        f.write('f score:' + str(f_score) + '\n')
        f.write('NDCG:' + str(NDCG) + '\n')
        f.write('\n')
        f.close()
# Claude vectorized eval

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
        f'../Final_results_loss/ml1m/min6/NN_results_userswithNproducts_split1.csv')
    results_metrics = pd.DataFrame.from_dict(
        data=metrics, orient='index').T[['precision', 'recall', 'f_score', 'revenue', 'ndcg']].to_numpy().flatten()

    np.savetxt(f'../Final_results_loss/ml1m/min6/NN_results_metrics_split1.csv',
               results_metrics.reshape(1, results_metrics.shape[0]),
               delimiter=",")

    return precision, recall, f_score, revenue, ndcg

# ta inja

def preprocessing(input_data, input_params):
    preprocessed_file_path = file_paths.preprocessed_dataset_file_name(
        input_params)

    try:
        # Only use the preprocessed data if the stored data type
        # is set to use pickle, otherwise always generate it on demand
        #
        if input_params['stored_data_type'] is not 'pickle':
            raise FileNotFoundError()

        with open(preprocessed_file_path, 'rb') as f:
            input_data = pickle.load(f)
            input_data['X_train'] = input_data['X_train'].toarray()
            input_data['X_test'] = input_data['X_test'].toarray()
            input_data['Y_train'] = input_data['Y_train'].toarray()
            input_data['Y_test'] = input_data['Y_test'].toarray()
            input_data['price_train'] = input_data['price_train'].toarray()
            input_data['price_test'] = input_data['price_test'].toarray()
            ds_preprocessing_time = 0
        print("Dataset already preprocessed, loading from {}".format(
            preprocessed_file_path))

    except FileNotFoundError:
        start_time = time.time()
        input_data = preprocessing__execution(input_data, input_params)
        print("back to preprocessing")
        file_paths.ensure_dir_exists(preprocessed_file_path)
        input_data['X_train'] = sparse.csr_matrix(input_data['X_train'])
        input_data['X_test'] = sparse.csr_matrix(input_data['X_test'])
        # these four are already made sparse in the preprocessing__execution func.
        input_data['Y_train'] = input_data['Y_train']
        input_data['Y_test'] = input_data['Y_test']
        input_data['price_train'] = input_data['price_train']
        input_data['price_test'] = input_data['price_test']

        pickle.dump(input_data, open(preprocessed_file_path, 'wb'))

        # store as sparse matrix, but return and continue as dense array for this array
        input_data['X_train'] = input_data['X_train'].toarray()
        input_data['X_test'] = input_data['X_test'].toarray()
        input_data['Y_train'] = input_data['Y_train'].toarray()
        input_data['Y_test'] = input_data['Y_test'].toarray()
        input_data['price_train'] = input_data['price_train'].toarray()
        input_data['price_test'] = input_data['price_test'].toarray()

        ds_preprocessing_time = time.time() - start_time
        print("Dataset preprocessing took:\t\t{}".format(ds_preprocessing_time))

    return input_data, ds_preprocessing_time


def preprocessing__execution(input_data, input_params):
    preprocessed_data = dict()
    X_train = input_data['X_train']
    X_test = input_data['X_test']
    Y_train = input_data['Y_train']
    Y_test = input_data['Y_test']

    """ Remove products that do not appear in the train data targets
    These can not be correctly predicted anyway
    sklearn often does not accept such data """

    all_products = Y_train.columns.to_numpy()

    bought_products = [
        product for product in all_products[Y_train.sum(axis=0) != 0]]

    # Split features into product and non-product features
    # to ensure that the product features are the last columns
    non_product_features_train = X_train.drop(all_products, axis=1)
    product_features_train = X_train[bought_products]

    non_product_features_test = X_test.drop(all_products, axis=1)
    product_features_test = X_test[bought_products]

    # One-hot encode the data scikit one-hot encoder
    #
    # lehl@2020-06-19: We might want to use sparse matrices here for bigger datasets (Insurance 1 and 2),
    # where we don't use toy sized datasets (MovieLens)
    #
    transformer = ColumnTransformer([
        ('OneHotEncoder', OneHotEncoder(sparse=False),
         input_params['categorical_columns'])
    ], remainder='passthrough')

    concat_series_X = pd.concat(
        [non_product_features_train, non_product_features_test])
    transformer.fit(concat_series_X)

    if input_params['printMatrices']:
        print("concat_series_X:\n", concat_series_X)

    non_product_features_train = transformer.transform(
        non_product_features_train)
    non_product_features_test = transformer.transform(
        non_product_features_test)

    if input_params['printMatrices']:
        print("non_product_features_train:\n", non_product_features_train)
        print("non_product_features_test:\n", non_product_features_test)

    preprocessed_data['X_train'] = np.concatenate(
        [non_product_features_train, product_features_train.to_numpy()], axis=1)
    preprocessed_data['X_test'] = np.concatenate(
        [non_product_features_test, product_features_test.to_numpy()], axis=1)
    print("in preprocessing execution")
    col_names = Y_train.columns.to_numpy()
    bought_products_idx = [np.where(col_names == c)[0].item()
                           for c in bought_products]

    preprocessed_data['Y_train'] = sparse. \
        csr_matrix(Y_train.values)[:, bought_products_idx]
    preprocessed_data['Y_test'] = sparse. \
        csr_matrix(Y_test.values)[:, bought_products_idx]

    preprocessed_data['products'] = bought_products

    if ('price_train' in input_data.keys()) and ('price_test' in input_data.keys()):
        # for memory issue, replace the code above with average price
        # store data to sparse matrix and then select bought products
        preprocessed_data['price_test'] = sparse. \
            csr_matrix(input_data['price_test'].values)[:, bought_products_idx]
        preprocessed_data['price_train'] = sparse. \
            csr_matrix(input_data['price_train'].values)[
            :, bought_products_idx]

    if input_params['printMatrices']:
        print("X_train:\n", preprocessed_data['X_train'])
        print("X_test:\n", preprocessed_data['X_test'])
        print("Y_train:\n", preprocessed_data['Y_train'])
        print("Y_test:\n", preprocessed_data['Y_test'])

    # Ensure column is not shuffled
    # Ensure features' last columns are product columns
    assert ((preprocessed_data['X_train'][:, -len(bought_products):] + preprocessed_data['Y_train']).max() <= 1.0).all()
    assert ((preprocessed_data['X_test'][:, -len(bought_products):] + preprocessed_data['Y_test']).max() <= 1.0).all()

    return preprocessed_data


def train__model_by_key(input_params, hyperparameter_defaults):
    """
    Multilearn classification Using BinaryRelevance for problem transfer
    """

    algorithm_key = input_params['algorithm_key']

    # transform reason:
    # some classifiers return the predictions in a slightly different format
    # e.g. a list of prediction for each product or a sparse matrix

    if algorithm_key == 'RandomForest':
        # transform = lambda x: x.toarray()
        def transform(x):
            if 'ndarray' in str(type(x)):
                return x
            else:
                return x.toarray()

        classifier = BinaryRelevance(
            classifier=RandomForestClassifier(n_estimators=10)
        )

    elif algorithm_key == 'DecisionTree':
        def transform(x):
            if 'ndarray' in str(type(x)):
                return x
            else:
                return x.toarray()

        classifier = BinaryRelevance(
            classifier=DecisionTreeClassifier(random_state=0)
        )

    elif algorithm_key == 'SVM':
        def transform(x):
            if 'ndarray' in str(type(x)):
                return x
            else:
                return x.toarray()

        classifier = BinaryRelevance(
            classifier=svm.SVC(gamma='scale')
        )

    elif algorithm_key == 'LogisticRegression':
        def transform(x):
            if 'ndarray' in str(type(x)):
                return x
            else:
                return x.toarray()

        classifier = BinaryRelevance(
            classifier=linear_model.LogisticRegression(solver='liblinear')
        )

    # using single model instead of binary relevance
    # similarly, they predict each product independently
    elif algorithm_key == 'NN':
        # lehl@2020-05-19:
        # Training the NN often brings up the error "convergence not reached",
        # as training stops after 200 iterations by default. Increasing the param
        # max_iter=1000 still yields the same warning and comparable results.
        #
        def transform(x):
            return x

        classifier = neural_network.MLPClassifier(
            verbose=True,
            hidden_layer_sizes=hyperparameter_defaults['hidden_layer_sizes'],
            activation=hyperparameter_defaults['activation'],
            solver=hyperparameter_defaults['solver'],
            alpha=hyperparameter_defaults['alpha'],
            batch_size=hyperparameter_defaults['batch_size'],
            learning_rate=hyperparameter_defaults['learning_rate'],
            learning_rate_init=hyperparameter_defaults['learning_rate_init'],
            early_stopping=hyperparameter_defaults['early_stopping'])
            # validation_fraction=hyperparameter_defaults['validation_fraction'])
            # max_iter=hyperparameter_defaults['max_iter'])
        print("random state:", classifier.random_state)

    elif algorithm_key == 'RF':
        def transform(x):
            return x[:, :, 1].T

        classifier = RandomForestClassifier(n_estimators=50)

    elif algorithm_key == 'DT':
        def transform(x):
            return x[:, :, 1].T

        classifier = DecisionTreeClassifier(random_state=0, max_depth=4)

    elif algorithm_key == 'LR':
        def transform(x):
            return x

        classifier = linear_model.LogisticRegression()

    elif(algorithm_key == 'CollaborativeFiltering'):
        def transform(x): return x

        try:
            classifier = CollaborativeFiltering(
                neighbor_ratio=input_params['cf_neighbor_ratio'])
        except KeyError:
            classifier = CollaborativeFiltering()

    elif algorithm_key == 'SPM':
        def transform(x):
            return x

        class SPM(object):
            def __init__(self):
                pass

            def fit(self, X, Y):
                # Ignores the products that the customer has
                # should improve the results in case of difference in
                # the distribution between the feature and target products,
                # e.g. with temporally ordered generation of the dataset
                self.frequencies = Y.sum(axis=0) / Y.shape[0]

            def predict_proba(self, X):
                item_scores = np.repeat([self.frequencies], X.shape[0], axis=0)
                user_scores = X.sum(axis=1, keepdims=True) / X.shape[1]
                return item_scores + user_scores

        classifier = SPM()

    elif algorithm_key == 'Baseline':
        def transform(x):
            return x

        class Baseline(object):
            def __init__(self):
                pass

            def fit(self, X, Y):
                self.frequencies = Y.sum(axis=0) / Y.sum()

            def predict_proba(self, X):
                return np.repeat([self.frequencies], X.shape[0], axis=0)

        classifier = Baseline()

    return classifier, transform

def binary_log_loss(y_true, y_prob):
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)

    return -(xlogy(y_true, y_prob).sum() +
             xlogy(1 - y_true, 1 - y_prob).sum()) / y_prob.shape[0]


def train(input_data, input_params, hyperparameter_defaults):
    # transform reason:
    # some classifiers return the predictions in a slightly different format
    # e.g. a list of prediction for each product or a sparse matrix

    classifier, transform = train__model_by_key(
        input_params, hyperparameter_defaults)
    # add minibatch
    NUM_EPOCHS = 100
    # NUM_PRODUCT = input_data['Y_train'].shape[1]
    # NUM_USERS = input_data['X_train'].shape[0]
    np.random.seed(42)
    train_losses = []
    test_losses = []
    f1_1 = []
    f1_2 = []
    f1_3 = []
    f1_4 = []
    f1_5 = []
    for e in range(NUM_EPOCHS):
        print('epoch:', e)
        classifier.partial_fit(
            input_data['X_train'], input_data['Y_train'], classes=np.arange(input_data['Y_train'].shape[1]))
        predict_proba = classifier.predict_proba(input_data['X_test'])
        train_loss = classifier.loss_
        test_loss = binary_log_loss(input_data['Y_test'], predict_proba)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_mask = input_data['Y_test']
        train_mask = \
            input_data['X_train'][:, :input_data['Y_train'].shape[1]] + \
            input_data['X_test'][:, :input_data['Y_train'].shape[1]] + \
            input_data['Y_train']
        price_mask = input_data['price_test']

        # precision, recall, f_score, revenue, ndcg = evaluate_jcastyle(
        #     predict_proba, test_mask, train_mask, price_mask)
        precision, recall, f_score, revenue, ndcg = test_model_all(
            predict_proba, test_mask, train_mask, price_mask)
        wandb_dict = {'loss': train_loss, 'test_loss': test_loss}
        for i, at_n in enumerate([1, 2, 3, 4, 5, 10, 15]):
            wandb_dict['f1_at' + str(at_n)] = f_score[i]
            wandb_dict['NDCG_at' + str(at_n)] = ndcg[i]
            wandb_dict['revenue_at' + str(at_n)] = revenue[i]
        wandb_log(wandb_dict, commit=True)

        # store result_metricss in csv done
        model_path = './metrics_results/' + str(wandb.run.group)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        import csv
        csv_dict = dict(
            {
                'wandb_group_name': wandb.run.group,
                'wandb_run_name': wandb.run.name,
                'epoch': wandb.run.summary['current_epoch']
            }, 
            **wandb_dict
        )
        csv_columns = ['wandb_group_name', 'wandb_run_name', 'epoch'] + list(sorted(wandb_dict.keys()))

        csv_file_path = model_path + '/metric_results.csv'
        if os.path.exists(csv_file_path) and os.path.isfile(csv_file_path):
            # Append to CSV
            # 
            with open(csv_file_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerow(csv_dict)
        else:
            # Create CSV
            # 
            with open(csv_file_path, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerow(csv_dict)
                
        # store result_metricss in csv done
        f1_1.append(f_score[0])
        f1_2.append(f_score[1])
        f1_3.append(f_score[2])
        f1_4.append(f_score[3])
        f1_5.append(f_score[4])

        print(f"[{e+1}]\tTRAIN: {train_loss}\tTEST: {test_loss}")
        print("ndcg:", ndcg)
        print("precision:", precision)
        print("recall:", recall)
        print("revenue:", revenue)
        print("f_score:", f_score)
        # evaliate epoch
        if e % 10 == 0:
            print("inside e % 10")
            start_training = time.time()
            training_time_taken = time.time() - start_training
            print("[{}] Training took\t\t{}".format(
                input_params['algorithm_key'], training_time_taken))
            print('epoch:', e)
            np.savetxt('../Final_results_loss/insurance/train_losses' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv', train_losses, delimiter=',')
            np.savetxt('../Final_results_loss/insurance/test_losses' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv', test_losses, delimiter=',')
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.ylim(bottom=0)
            plt.legend()
            plt.title("Train and Test loss of MLP method over epochs on insurance dataset.")
            plt.savefig('../Final_results_loss/insurance/insurance' +
                        str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.png')
            plt.close()
            plt.figure()
            plt.plot(f1_1, label='F1@1')
            plt.plot(f1_2, label='F1@2')
            plt.plot(f1_3, label='F1@3')
            plt.plot(f1_4, label='F1@4')
            plt.plot(f1_5, label='F1@5')
            plt.ylim(bottom=0)
            plt.legend()
            plt.title("F1@N for MLP method over epochs on insurance dataset")
            time.sleep(5)
            plt.savefig('../Final_results_loss/insurance/insurance' +
                        str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '.png')
            plt.close()
    return classifier, transform, training_time_taken

# Normalizes the given prediction probabiltiies
#


def normalize_predictions_probabilities(predictions_proba):
    normalized_proba = predictions_proba.copy()

    # lehl@2020-05-19:
    # It can be that the classifier predicts 0 for all values of a user (is this a bug in RF?)
    # --> Set those to 1/num_products for all values in that row
    #
    zero_customer_idx = np.where(np.sum(normalized_proba, axis=1) == 0)
    normalized_proba[zero_customer_idx, :] = 1.0

    # lehl@2020-05-13:
    # Normalize the prediction probabilities such that for each
    # customer the probabilities sum up to 1
    #
    normalized_proba = normalized_proba / \
        np.sum(normalized_proba, axis=1)[:, np.newaxis]

    # Check that all customers sum up to 1 (+- 10^-5 / 10^-8)
    #
    assert(np.isclose(np.sum(normalized_proba, axis=1), 1).all())

    return normalized_proba
