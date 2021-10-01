
import pandas as pd
import numpy as np


class ml1m:
    def __init__(self, split_num):
        self.split_num = str(split_num)
        self.data_path = (
            './data/ml-1m/max5' + self.split_num + '/')

    @staticmethod
    def train(self, n):
        train_df = pd.read_csv(
            self.data_path + 'train.csv').drop(columns=['price'])
        vali_df = pd.read_csv(
            self.data_path + 'vali.csv').drop(columns=['price'])
        test_df = pd.read_csv(
            self.data_path + 'test.csv').drop(columns=['price'])
        train_df['rating'] = 1
        train_R_matrix = pd.pivot_table(data=train_df,
                                        index='userId',
                                        columns='itemId',
                                        values='rating',
                                        fill_value=0)

        vali_df['rating'] = 1
        vali_R_matrix = pd.pivot_table(data=vali_df,
                                       index='userId',
                                       columns='itemId',
                                       values='rating',
                                       fill_value=0)

        unique_item_names = list(
            set(list(train_df['itemId']) + list(test_df['itemId'])))
        unique_user_names = list(
            set(list(train_df['userId']) + list(test_df['userId'])))

        # add missing items
        train_R_matrix = train_R_matrix.reindex(columns=train_R_matrix.columns.to_list() +
                                                list(
                                                    set(unique_item_names) - set(train_R_matrix.columns.to_list())),
                                                fill_value=0)
        vali_R_matrix = vali_R_matrix.reindex(columns=vali_R_matrix.columns.to_list() +
                                              list(
                                                  set(unique_item_names) - set(vali_R_matrix.columns.to_list())),
                                              fill_value=0)

        # add missing users
        # to test
        vali_missing_users_df = pd.DataFrame([[0] * vali_R_matrix.shape[1]], columns=vali_R_matrix.columns,
                                             index=list(set(unique_user_names) - set(vali_R_matrix.index)))
        vali_R_matrix = vali_R_matrix.append(
            vali_missing_users_df).sort_index()

        # to train
        train_missing_users_df = pd.DataFrame([[0] * train_R_matrix.shape[1]], columns=train_R_matrix.columns,
                                              index=list(set(unique_user_names) - set(train_R_matrix.index)))
        train_R_matrix = train_R_matrix.append(
            train_missing_users_df).sort_index()

        # sort items
        train_R_matrix = train_R_matrix.sort_index(axis=1)
        vali_R_matrix = vali_R_matrix.sort_index(axis=1)

        train_R = train_R_matrix.to_numpy()
        vali_R = vali_R_matrix.to_numpy()

        return train_R, vali_R

    def test(self):
        print("in data preprocessing.", self.split_num)
        test_df = pd.read_csv(
            self.data_path + 'test.csv').drop(columns=['price'])

        test_df['rating'] = 1
        test_R_matrix = pd.pivot_table(data=test_df,
                                       index='userId',
                                       columns='itemId',
                                       values='rating',
                                       fill_value=0)

        train_df = pd.read_csv(
            self.data_path + 'train.csv').drop(columns=['price'])

        train_df['rating'] = 1
        train_R_matrix = pd.pivot_table(data=train_df,
                                        index='userId',
                                        columns='itemId',
                                        values='rating',
                                        fill_value=0)

        train_price_df = pd.read_csv(self.data_path + 'train.csv')
        train_price_matrix = pd.pivot_table(data=train_price_df,
                                            index='userId',
                                            columns='itemId',
                                            values='price',
                                            fill_value=0)

        test_price_df = pd.read_csv(self.data_path + 'test.csv')
        test_price_matrix = pd.pivot_table(data=test_price_df,
                                           index='userId',
                                           columns='itemId',
                                           values='price',
                                           fill_value=0)

        # add missing items
        unique_item_names = list(
            set(list(train_df['itemId']) + list(test_df['itemId'])))
        train_R_matrix = train_R_matrix.reindex(columns=train_R_matrix.columns.to_list() +
                                                list(
                                                    set(unique_item_names) - set(train_R_matrix.columns.to_list())),
                                                fill_value=0)
        test_R_matrix = test_R_matrix.reindex(columns=test_R_matrix.columns.to_list() +
                                              list(
                                                  set(unique_item_names) - set(test_R_matrix.columns.to_list())),
                                              fill_value=0)

        train_price_matrix = train_price_matrix.reindex(columns=train_price_matrix.columns.to_list() +
                                                        list(
                                                            set(unique_item_names) - set(train_price_matrix.columns.to_list())),
                                                        fill_value=0)

        test_price_matrix = test_price_matrix.reindex(columns=test_price_matrix.columns.to_list() +
                                                      list(
                                                          set(unique_item_names) - set(test_price_matrix.columns.to_list())),
                                                      fill_value=0)

        # add missing users
        unique_user_names = list(
            set(list(train_df['userId']) + list(test_df['userId'])))
        # to test
        test_missing_users_df = pd.DataFrame([[0] * test_R_matrix.shape[1]], columns=test_R_matrix.columns,
                                             index=list(set(unique_user_names) - set(test_R_matrix.index)))
        test_R_matrix = test_R_matrix.append(
            test_missing_users_df).sort_index()

        # to train
        train_missing_users_df = pd.DataFrame([[0] * train_R_matrix.shape[1]], columns=train_R_matrix.columns,
                                              index=list(set(unique_user_names) - set(train_R_matrix.index)))
        train_R_matrix = train_R_matrix.append(
            train_missing_users_df).sort_index()

        # to train price
        train_price_missing_users_df = pd.DataFrame([[0] * train_price_matrix.shape[1]], columns=train_price_matrix.columns,
                                                    index=list(set(unique_user_names) - set(train_price_matrix.index)))
        train_price_matrix = train_price_matrix.append(
            train_price_missing_users_df).sort_index()

        # to test price
        test_price_missing_users_df = pd.DataFrame([[0] * test_price_matrix.shape[1]], columns=test_price_matrix.columns,
                                                   index=list(set(unique_user_names) - set(test_price_matrix.index)))
        test_price_matrix = test_price_matrix.append(
            test_price_missing_users_df).sort_index()

        # sort items
        train_R_matrix = train_R_matrix.sort_index(axis=1)
        test_R_matrix = test_R_matrix.sort_index(axis=1)
        train_price_matrix = train_price_matrix.sort_index(axis=1)
        test_price_matrix = test_price_matrix.sort_index(axis=1)

        train_R = train_R_matrix.to_numpy()
        test_R = test_R_matrix.to_numpy()
        train_price_R = train_price_matrix.to_numpy()
        test_price_R = test_price_matrix.to_numpy()

        print("in preprocessing.")
        # average the price value 
        # avg train
        train_price_R = np.ma.masked_equal(
            train_price_R, 0).mean(axis=0).filled(0)
        ones = np.ones(train_R.shape)
        train_price_R = np.multiply(ones, train_price_R, dtype="float32")
        # avg test
        test_price_R = np.ma.masked_equal(
            test_price_R, 0).mean(axis=0).filled(0)
        ones = np.ones(test_R.shape)
        test_price_R = np.multiply(ones, test_price_R, dtype="float32")

        return train_R, test_R, train_price_R, test_price_R
