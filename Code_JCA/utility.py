from __future__ import division

import datetime
from math import log
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
import time
from openpyxl import load_workbook, Workbook
from wandb_decorator import *

AT_NS = np.array([1, 2, 3, 4, 5, 10, 15])


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


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test, products_prices=None):
    dcg_list = []
    count_1, count_2, count_3, count_4, count_5, count_10, count_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    revenue_1, revenue_2, revenue_3, revenue_4, revenue_5, revenue_10, revenue_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # compute the number of true positive items at top k
    for i in xrange(15):
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

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_2, precision_3, precision_4, precision_5, precision_10, precision_15]),\
        np.array([recall_1, recall_2, recall_3, recall_4, recall_5, recall_10, recall_15]), \
        np.array([ndcg_tmp_1, ndcg_tmp_2, ndcg_tmp_3, ndcg_tmp_4, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15]), \
        np.array([revenue_1, revenue_2, revenue_3, revenue_4,
                  revenue_5, revenue_10, revenue_15])


@wandb_timing
def dcg_at_k__numpy(r, k):
    """Score is discounted cumulative gain (dcg) Relevance is positive real values. Can use binary as the previous methods.
    Based on https://stats.stackexchange.com/questions/341611/proper-way-to-use-ndcgk-score-for-recommendations
    Args:
        r: Relevance scores (list or numpy) in rank order (first element is the first item)
        k: Number of results to consider
        (weights are [1.0, 0.6309, 0.5, 0.4307, ...])
    """
    r = np.asfarray(r)[:k]
    if r.size:
        # return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


@wandb_timing
def ndcg_at_k__numpy(r, k=20):
    """Score is normalized discounted cumulative gain (ndcg) Relevance is positive real values. Can use binary as the previous methods.
    Based on https://stats.stackexchange.com/questions/341611/proper-way-to-use-ndcgk-score-for-recommendations
    Args:
        r: Relevance scores (list or numpy) in rank order (first element is the first item)
        k: Number of results to consider
        (weights are [1.0, 0.6309, 0.5, 0.4307, ...])
    """
    dcg_max = dcg_at_k__numpy(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k__numpy(r, k) / dcg_max

# lehl@2021-04-28: Vectorized variant of the "user_precision_recall_ndcg" method,
# making use of numpy's execution speed
#


@wandb_timing
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

# calculate the metrics of the result


@wandb_timing
def test_model_all(prediction, test_mask, train_mask, price_mask, store_intermediate='no_store'):
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
    c = datetime.datetime.now()
    print("Collecting and saving the prediction data took: ", c - b)

    return precision, recall, f_score, ndcg, revenue, None


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


def get_train_instances(train_R, neg_sample_rate):
    """
    genderate training dataset for NCF models in each iteration
    :param train_R:
    :param neg_sample_rate:
    :return:
    """
    # randomly sample negative samples
    mask = neg_sampling(train_R, range(train_R.shape[0]), neg_sample_rate)

    user_input, item_input, labels = [], [], []
    idx = np.array(np.where(mask == 1))
    for i in range(idx.shape[1]):
        # positive instance
        u_i = idx[0, i]
        i_i = idx[1, i]
        user_input.append(u_i)
        item_input.append(i_i)
        labels.append(train_R[u_i, i_i])
    return user_input, item_input, labels


def neg_sampling(train_R, idx, neg_sample_rate):
    """
    randomly negative smaples
    :param train_R:
    :param idx:
    :param neg_sample_rate:
    :return:
    """
    num_cols = train_R.shape[1]
    num_rows = train_R.shape[0]
    # randomly sample negative samples
    mask = copy.copy(train_R)
    if neg_sample_rate == 0:
        return mask
    for b_idx in idx:
        mask_list = mask[b_idx, :]
        unobsv_list = np.where(mask_list == 0)
        unobsv_list = unobsv_list[0]  # unobserved indices
        obsv_num = num_cols - len(unobsv_list)
        neg_num = int(obsv_num * neg_sample_rate)
        # if the observed positive ratings are more than the half
        if neg_num > len(unobsv_list):
            neg_num = len(unobsv_list)
        if neg_num == 0:
            neg_num = 1
        neg_samp_list = np.random.choice(
            unobsv_list, size=neg_num, replace=False)
        mask_list[neg_samp_list] = 1
        mask[b_idx, :] = mask_list
    return mask


def pairwise_neg_sampling(train_R, r_idx, c_idx, neg_sample_rate):
    # import code; code.interact(local=dict(globals(), **locals()))
    R = train_R[r_idx, :]
    R = R[:, c_idx]
    p_input, n_input = [], []
    # it was (R == 1), we changed to !=0 because the train
    # smatrix now has prices.
    obsv_list = np.where(R != 0)

    unobsv_mat = []
    for r in range(R.shape[0]):
        unobsv_list = np.where(R[r, :] == 0)
        unobsv_list = unobsv_list[0]
        unobsv_mat.append(unobsv_list)

    for i in range(len(obsv_list[1])):
        # positive instance
        u = obsv_list[0][i]
        # negative instances
        unobsv_list = unobsv_mat[u]
        neg_samp_list = np.random.choice(
            unobsv_list, size=neg_sample_rate, replace=False)
        # import code; code.interact(local=dict(globals(), **locals()))
        for ns in neg_samp_list:
            p_input.append([u, obsv_list[1][i]])
            n_input.append([u, ns])
    # print('dataset size = ' + str(len(p_input)))
    return np.array(p_input), np.array(n_input)


# calculate the metrics of the result
def test_model_batch(prediction, test_mask, train_mask):
    precision_1, precision_5, precision_10, precision_15 = 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_5, recall_10, recall_15 = 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000
    precision = np.array(
        [precision_1, precision_5, precision_10, precision_15])
    recall = np.array([recall_1, recall_5, recall_10, recall_15])
    ndcg = np.array([ndcg_1, ndcg_5, ndcg_10, ndcg_15])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        # the indices of the true positive items in the test set
        u_test = np.where(u_test == 1)[0]
        u_pred = prediction[u, :]

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train,
                           u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u, revenue_u = user_precision_recall_ndcg(
                top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1

    return precision, recall, ndcg

# calculate the metrics of the result


def test_model_cold_start(prediction, test_mask, train_mask):
    precision_1, precision_5, precision_10, precision_15 = 0.0000, 0.0000, 0.0000, 0.0000
    recall_1, recall_5, recall_10, recall_15 = 0.0000, 0.0000, 0.0000, 0.0000
    ndcg_1, ndcg_5, ndcg_10, ndcg_15 = 0.0000, 0.0000, 0.0000, 0.0000
    precision = np.array(
        [precision_1, precision_5, precision_10, precision_15])
    recall = np.array([recall_1, recall_5, recall_10, recall_15])
    ndcg = np.array([ndcg_1, ndcg_5, ndcg_10, ndcg_15])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    n = 0
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        # the indices of the true positive items in the test set
        u_test = np.where(u_test == 1)[0]
        if len(u_test) > 10:
            continue
        u_pred = prediction[u, :]

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train,
                           u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u, revenue_u = user_precision_recall_ndcg(
                top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
            n += 1

    # compute the average over all users
    precision /= n
    recall /= n
    ndcg /= n
    print('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]'
          % (precision[0],
             precision[1],
             precision[2],
             precision[3]))
    print('recall_1   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]'
          % (recall[0], recall[1],
             recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[
        3] == 0 else 0
    print('f_measure_1\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]'
          % (f_measure_1,
             f_measure_5,
             f_measure_10,
             f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print('ndcg_1     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]'
          % (ndcg[0],
             ndcg[1],
             ndcg[2],
             ndcg[3]))
    return precision, recall, f_score, ndcg


def test_model_factor(prediction, test_mask, train_mask):
    item_list = np.zeros(train_mask.shape[1])
    item_list_rank = np.zeros(train_mask.shape[1])

    prediction = prediction + train_mask * -100000.0

    user_num = prediction.shape[0]
    for u in range(user_num):  # iterate each user
        u_test = test_mask[u, :]
        # the indices of the true positive items in the test set
        u_test = np.where(u_test == 1)[0]
        len_u_test = len(u_test)
        u_pred = prediction[u, :]

        top10_item_idx_no_train = np.argpartition(u_pred, -10)[-10:]
        item_list[top10_item_idx_no_train] += 1
        for i in range(len(top10_item_idx_no_train)):
            item_list_rank[top10_item_idx_no_train[i]] += (10 - i)

    item_count = np.sum(train_mask, axis=0)
    df = pd.DataFrame({'item_pred_freq': item_list, 'item_count': item_count})
    df.to_csv('data/no-factor' +
              time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) + '.csv')
    df = pd.DataFrame(
        {'item_pred_rank': item_list_rank, 'item_count': item_count})
    df.to_csv('data/rank-no-factor' +
              time.strftime('%y-%m-%d-%H-%M-%S', time.localtime()) + '.csv')
