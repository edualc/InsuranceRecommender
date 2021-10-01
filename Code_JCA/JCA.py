from wandb_decorator import *
import wandb
import datetime
import matplotlib.pyplot as plt
from pareto_manager_class import ParetoManager
from scipy import sparse
import utility
import copy
import matplotlib
import os
import pandas as pd
import numpy as np
import time
import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
matplotlib.use('agg')


class JCA:

    def __init__(self, sess, args, train_R, vali_R, train_price_R, vali_price_R, metric_path, date, data_name,
                 result_path=None):

        if args.f_act == "Sigmoid":
            f_act = tf.nn.sigmoid
        elif args.f_act == "Relu":
            f_act = tf.nn.relu
        elif args.f_act == "Tanh":
            f_act = tf.nn.tanh
        elif args.f_act == "Identity":
            f_act = tf.identity
        elif args.f_act == "Elu":
            f_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        if args.g_act == "Sigmoid":
            g_act = tf.nn.sigmoid
        elif args.g_act == "Relu":
            g_act = tf.nn.relu
        elif args.g_act == "Tanh":
            g_act = tf.nn.tanh
        elif args.g_act == "Identity":
            g_act = tf.identity
        elif args.g_act == "Elu":
            g_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        self.sess = sess
        self.args = args

        self.base = args.base

        self.num_rows = train_R.shape[0]
        self.num_cols = train_R.shape[1]
        self.U_hidden_neuron = args.U_hidden_neuron
        self.I_hidden_neuron = args.I_hidden_neuron

        self.train_R = train_R
        self.vali_R = vali_R

        self.train_price_R = train_price_R
        self.vali_price_R = vali_price_R
        self.num_test_ratings = np.sum(vali_R)

        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch_U = int(self.num_rows / float(self.batch_size)) + 1
        self.num_batch_I = int(self.num_cols / float(self.batch_size)) + 1

        self.lr = args.lr  # learning rate
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.margin = args.margin

        self.f_act = f_act  # the activation function for the output layer
        self.g_act = g_act  # the activation function for the hidden layer

        self.global_step = tf.Variable(0, trainable=False)

        self.lambda_value = args.lambda_value  # regularization term trade-off

        self.result_path = result_path
        self.metric_path = metric_path
        self.date = date  # today's date
        self.data_name = data_name
        # @esla Multiobj
        self.losses_names = args.losses_names

        self.neg_sample_rate = args.neg_sample_rate

        self.model_path = None

        self.U_OH_mat = sparse.eye(self.num_rows, dtype=float)
        self.I_OH_mat = np.eye(self.num_cols, dtype=float)

        # pareto manager
        self.pareto_manager = ParetoManager(
            self.sess, self.args, PATH='./pareto_models/models/')

        print('**********JCA**********')
        print(self.args)
        self.prepare_model()

    def run(self, train_R, vali_R, train_price_R, vali_price_R):
        print("in JCA run")

        self.train_R = train_R
        self.vali_R = vali_R
        self.train_price_R = train_price_R
        self.vali_price_R = vali_price_R

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            self.save_model()
        else:
            self.load_model()

        if len(self.losses_names) >= 2:
            self.max_losses_np = self.compute_max_empirical_losses()
        elif len(self.losses_names) == 1:
            self.max_losses_np = [0.0]
        train_losses = []
        test_losses = []
        ndcg_1 = []
        ndcg_2 = []
        ndcg_3 = []
        ndcg_4 = []
        ndcg_5 = []
        f1_1 = []
        f1_2 = []
        f1_3 = []
        f1_4 = []
        f1_5 = []
        rev_1 = []
        rev_2 = []
        rev_3 = []
        rev_4 = []
        rev_5 = []
        for epoch_itr in xrange(self.train_epoch):
            train_loss, test_loss, pos_train, pos_test, metrics = self.train_model(
                epoch_itr)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            ndcg_1.append(metrics['NDCG'][0])
            ndcg_2.append(metrics['NDCG'][1])
            ndcg_3.append(metrics['NDCG'][2])
            ndcg_4.append(metrics['NDCG'][3])
            ndcg_5.append(metrics['NDCG'][4])
            f1_1.append(metrics['f1'][0])
            f1_2.append(metrics['f1'][1])
            f1_3.append(metrics['f1'][2])
            f1_4.append(metrics['f1'][3])
            f1_5.append(metrics['f1'][4])
            rev_1.append(metrics['revenue'][0])
            rev_2.append(metrics['revenue'][1])
            rev_3.append(metrics['revenue'][2])
            rev_4.append(metrics['revenue'][3])
            rev_5.append(metrics['revenue'][4])
            revenue = [rev_1, rev_2, rev_3, rev_4, rev_5]
            f_score = [f1_1, f1_2, f1_3, f1_4, f1_5]
            NDCG = [ndcg_1, ndcg_2, ndcg_3, ndcg_4, ndcg_5]
            metrics = {'f1': f_score, 'NDCG': NDCG, 'revenue': revenue}

            self.save_model()

            # if epoch_itr % 1 == 0:
            # self.test_model(epoch_itr)
        # return self.make_records()
    def _make_subpath(self):
        loss_names = ''
        for loss in self.args.losses_names:
            loss_names = loss_names + loss + '_'
        # make a directory with model params if not exist.
        subdir_path = 'batch' + str(self.args.batch_size) + '_' + \
            str(self.args.method) + '_' + \
            'lr' + str(self.args.lr) + '_' + \
            'lambda' + str(self.args.lambda_value) + '_' + \
            'opt' + str(self.args.optimizer_method) + '_' + \
            'obj' + str(loss_names) + \
            'split' + str(self.args.split_num)

        return subdir_path

    @wandb_timing
    def compute_max_empirical_losses(self):
        # @Esla Multiobj:
        """A helper function for approximating the maximum empirical loss the model
        could have. It is called by jca constructor __init__ function.
        """
        # approximate the max loss empirically
        print("calculating empirical loss")
        random_row_idx = np.random.permutation(
            self.num_rows)  # randomly permute the rows
        random_col_idx = np.random.permutation(
            self.num_cols)  # randomly permute the cols
        self.max_losses_np = [0.0, 0.0]

        for i in xrange(self.num_batch_U):  # iterate each batch
            if i == self.num_batch_U - 1:
                row_idx = random_row_idx[i * self.batch_size:]
            else:
                row_idx = random_row_idx[(
                    i * self.batch_size):((i + 1) * self.batch_size)]
            for j in xrange(self.num_batch_I):
                # get the indices of the current batch
                if j == self.num_batch_I - 1:
                    col_idx = random_col_idx[j * self.batch_size:]
                else:
                    col_idx = random_col_idx[(
                        j * self.batch_size):((j + 1) * self.batch_size)]
                p_input, n_input = utility.pairwise_neg_sampling(
                    self.train_R, row_idx, col_idx, self.neg_sample_rate)

                input_R_U = self.train_R[row_idx, :]
                input_R_I = self.train_R[:, col_idx]
                price_R_U_train = self.train_price_R[row_idx, :]
                price_R_I_train = self.train_price_R[:, col_idx]
                # ---------------------------------------------------------------------
                max_losses = self.sess.run(
                    [self.cost_correctness, self.cost_revenue],
                    feed_dict={
                        self.input_R_U: input_R_U,
                        self.input_R_I: input_R_I,
                        self.input_OH_I: self.I_OH_mat[col_idx, :],
                        self.input_P_cor: p_input,
                        self.input_N_cor: n_input,
                        self.input_price_R_U: price_R_U_train,
                        self.input_price_R_I: price_R_I_train,
                        self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                        self.col_idx: np.reshape(col_idx, (len(col_idx), 1))})
                # total batch max impirical loss
                self.max_losses_np = np.add(self.max_losses_np, max_losses)
        # ---------------------------------------------------------------------
        # finish impirical loss calculation
        return self.max_losses_np

    @wandb_timing
    def get_gradient_np(self, grad_val_tuple):
        """A helper function for obtaining the gradients of the model in a numpy
        array."""
        gradient = []
        print("flat the gradients")
        try:
            for grad, val in grad_val_tuple:
                gradient.append(tf.reshape(grad, [-1]))

            return np.concatenate(gradient)
        except Exception:
            size = 0
            for grad, val in grad_val_tuple:
                size += tf.reshape(grad, [-1]).shape[0].value

            return np.zeros(shape=size)

    def prepare_model(self):

        # input rating vector
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[
                                        None, self.num_cols], name="input_R_U")
        self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[
                                        self.num_rows, None], name="input_R_I")
        self.input_OH_I = tf.placeholder(dtype=tf.float32, shape=[
                                         None, self.num_cols], name="input_OH_I")
        self.input_P_cor = tf.placeholder(
            dtype=tf.int32, shape=[None, 2], name="input_P_cor")
        self.input_N_cor = tf.placeholder(
            dtype=tf.int32, shape=[None, 2], name="input_N_cor")
        self.input_price_R_U = tf.placeholder(
            dtype=tf.float32, shape=[None, self.num_cols], name="input_price_R_U")
        self.input_price_R_I = tf.placeholder(
            dtype=tf.float32, shape=[self.num_rows, None], name="input_price_R_I")

        # input indicator vector indicator
        self.row_idx = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name="row_idx")
        self.col_idx = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name="col_idx")

        # maximum impirical loss
        # --------------------------------------------------------
        self.max_losses = tf.placeholder(dtype=tf.float32, shape=[
                                         1, len(self.losses_names)], name="max_losses")
        # --------------------------------------------------------

        # user component
        # first layer weights
        UV = tf.get_variable(name="UV", initializer=tf.truncated_normal(shape=[self.num_cols, self.U_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer weights
        UW = tf.get_variable(name="UW", initializer=tf.truncated_normal(shape=[self.U_hidden_neuron, self.num_cols],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # first layer bias
        Ub1 = tf.get_variable(name="Ub1", initializer=tf.truncated_normal(shape=[1, self.U_hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)
        # second layer bias
        Ub2 = tf.get_variable(name="Ub2", initializer=tf.truncated_normal(shape=[1, self.num_cols],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)

        # item component
        # first layer weights
        IV = tf.get_variable(name="IV", initializer=tf.truncated_normal(shape=[self.num_rows, self.I_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer weights
        IW = tf.get_variable(name="IW", initializer=tf.truncated_normal(shape=[self.I_hidden_neuron, self.num_rows],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # first layer bias
        Ib1 = tf.get_variable(name="Ib1", initializer=tf.truncated_normal(shape=[1, self.I_hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)
        # second layer bias
        Ib2 = tf.get_variable(name="Ib2", initializer=tf.truncated_normal(shape=[1, self.num_rows],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)

        I_factor_vector = tf.get_variable(name="I_factor_vector", initializer=tf.random_uniform(shape=[1, self.num_cols]),
                                          dtype=tf.float32)

        # user component
        # input to the hidden layer
        U_pre_Encoder = tf.matmul(self.input_R_U, UV) + Ub1
        # output of the hidden layer
        self.U_Encoder = self.g_act(U_pre_Encoder)
        # input to the output layer
        U_pre_Decoder = tf.matmul(self.U_Encoder, UW) + Ub2
        # output of the output layer
        self.U_Decoder = self.f_act(U_pre_Decoder)

        # item component
        I_pre_mul = tf.transpose(
            tf.matmul(I_factor_vector, tf.transpose(self.input_OH_I)))
        # input to the hidden layer
        I_pre_Encoder = tf.matmul(tf.transpose(self.input_R_I), IV) + Ib1
        # output of the hidden layer
        self.I_Encoder = self.g_act(I_pre_Encoder * I_pre_mul)
        # input to the output layer
        I_pre_Decoder = tf.matmul(self.I_Encoder, IW) + Ib2
        # output of the output layer
        self.I_Decoder = self.f_act(I_pre_Decoder)

        # final output
        self.Decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.U_Decoder), self.col_idx)))
                        + tf.gather_nd(tf.transpose(self.I_Decoder), self.row_idx)) / 2.0

        # @Esla: Multiobj:
        # cost 2: regularization which is same for both losses.
        pre_cost2 = tf.square(self.l2_norm(UW)) + tf.square(self.l2_norm(UV)) \
            + tf.square(self.l2_norm(IW)) + tf.square(self.l2_norm(IV))\
            + tf.square(self.l2_norm(Ib1)) + tf.square(self.l2_norm(Ib2))\
            + tf.square(self.l2_norm(Ub1)) + tf.square(self.l2_norm(Ub2))
        cost2 = self.lambda_value * 0.5 * pre_cost2  # regularization term

        pre_cost1s = [0.0] * len(self.losses_names)
        self.cost1s = [0.0] * len(self.losses_names)
        self.cost = [0.0] * len(self.losses_names)
        for i_l, loss in enumerate(self.losses_names):
            # compute the cost for each loss.
            pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
            neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)

            if loss == 'correctness_loss':
                print("correctness loss")
            elif loss == 'revenue_loss':
                print("revenue loss")
                pos_data_price = tf.math.log(tf.math.add(tf.gather_nd(
                    self.input_price_R_U, self.input_P_cor), tf.constant(1, dtype=tf.float32)))
                pos_data = pos_data * pos_data_price

                neg_data_price = tf.math.log(tf.math.add(tf.gather_nd(
                    self.input_price_R_U, self.input_N_cor), tf.constant(1, dtype=tf.float32)))
                neg_data = neg_data * neg_data_price

            pre_cost1s[i_l] = tf.maximum(neg_data - pos_data + self.margin,
                                         tf.zeros(tf.shape(neg_data)[0]))

            self.cost1s[i_l] = tf.reduce_sum(
                pre_cost1s[i_l])  # prediction squared error

        if self.losses_names == ['correctness_loss']:
            print("only one objective: correctness")
            self.cost_final = self.cost1s[0] + cost2

        elif self.losses_names == ['revenue_loss']:
            print("only one objective: revenue")
            self.cost_final = self.cost1s[0] + cost2

        elif self.losses_names == ['correctness_loss', 'revenue_loss']:
            print("two objectives: correctness and revenue")

            for i_l, loss in enumerate(self.losses_names):
                if loss == 'correctness_loss':
                    self.cost_correctness = self.cost1s[i_l]
                elif loss == 'revenue_loss':
                    self.cost_revenue = self.cost1s[i_l]

                # + cost2  todo ! add cost2 after normalizing the revneue cost.
                self.cost[i_l] = (self.cost1s[i_l])

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        for i_l, loss in enumerate(self.losses_names):
            if loss == 'correctness_loss':
                if len(self.losses_names) >= 2:
                    # compute gradient
                    print("do nothing in >=2")
                    self.gvs_correctness = optimizer.compute_gradients(
                        self.cost_correctness)
                    self.gvs_correctness = self.get_gradient_np(
                        self.gvs_correctness)
                    # # normalize gradient
                    self.gvs_correctness /= self.max_losses[0, i_l]
                elif len(self.losses_names) == 1:
                    # if only one objective
                    # compute gradient
                    self.gvs_correctness = optimizer.compute_gradients(
                        self.cost_final)
                    self.optimizer = optimizer.apply_gradients(
                        self.gvs_correctness, global_step=self.global_step)

            if loss == 'revenue_loss':
                if len(self.losses_names) >= 2:
                    print("do nothing in >=2")
                    # # compute gradient
                    self.gvs_revenue = optimizer.compute_gradients(
                        self.cost_revenue)
                    self.gvs_revenue = self.get_gradient_np(self.gvs_revenue)
                    # # normalize gradients
                    self.gvs_revenue /= self.max_losses[0, i_l]
                elif len(self.losses_names) == 1:
                    # if only one objective
                    # compute gradient
                    self.gvs_revenue = optimizer.compute_gradients(
                        self.cost_final)
                    self.optimizer = optimizer.apply_gradients(
                        self.gvs_revenue, global_step=self.global_step)

        if len(self.losses_names) == 2:
            @wandb_timing
            def calculate_alpha():
                # QCOP Solver
                # compute alphas for two objectives
                self.alpha = tf.tensordot(tf.subtract(self.gvs_correctness, self.gvs_revenue), self.gvs_correctness, axes=1) \
                    / tf.tensordot(tf.subtract(self.gvs_revenue, self.gvs_correctness), tf.subtract(self.gvs_revenue, self.gvs_correctness), axes=1)
                self.alpha = tf.clip_by_value(
                    self.alpha, clip_value_min=0, clip_value_max=1)

                return [[self.alpha, 1-self.alpha]]

            self.alphas = tf.cond(tf.reduce_all(tf.equal(
                self.gvs_correctness, self.gvs_revenue)), lambda: [0.5, 0.5], lambda: calculate_alpha())

            for i_l in range(len(self.losses_names)):
                if i_l == 0:
                    # 0.5 * self.cost[i_l]  / self.max_losses[0,i_l]
                    self.cost_final = self.alphas[i_l] * \
                        (self.cost[i_l] / self.max_losses[0, i_l])
                else:
                    # 0.5 * self.cost[i_l]  / self.max_losses[0,i_l]
                    self.cost_final += self.alphas[i_l] * \
                        (self.cost[i_l] / self.max_losses[0, i_l])

            self.gvs = optimizer.compute_gradients(self.cost_final)
            self.optimizer = optimizer.apply_gradients(
                self.gvs, global_step=self.global_step)
            print("end model preparation.")

    @wandb_timing__end_epoch
    def train_model(self, itr):

        random_row_idx = np.random.permutation(
            self.num_rows)  # randomly permute the rows
        random_col_idx = np.random.permutation(
            self.num_cols)  # randomly permute the cols
        batch_cost = 0.0
        batch_cost_test = 0.0
        pos_train = 0
        pos_test = 0
        start_time = time.time()
        ts = 0

        t_start__batch_training = time.time()
        # Start of the normal training.
        print('Starting to iterate over ' +
              str(self.num_batch_U) + ' U batches...')
        for i in xrange(self.num_batch_U):  # iterate each batch
            if ((i % 100) == 0):
                print('i: ', i)
            if i == self.num_batch_U - 1:
                row_idx = random_row_idx[i * self.batch_size:]
            else:
                row_idx = random_row_idx[(
                    i * self.batch_size):((i + 1) * self.batch_size)]
            for j in xrange(self.num_batch_I):
                # get the indices of the current batch
                if j == self.num_batch_I - 1:
                    col_idx = random_col_idx[j * self.batch_size:]
                else:
                    col_idx = random_col_idx[(
                        j * self.batch_size):((j + 1) * self.batch_size)]
                ts1 = time.time()
                p_input, n_input = utility.pairwise_neg_sampling(
                    self.train_R, row_idx, col_idx, self.neg_sample_rate)
                p_input_test, n_input_test = utility.pairwise_neg_sampling(
                    self.vali_R, row_idx, col_idx, self.neg_sample_rate)
                ts2 = time.time()
                ts += (ts2 - ts1)

                input_R_U = self.train_R[row_idx, :]
                input_R_I = self.train_R[:, col_idx]
                input_R_U_test = self.vali_R[row_idx, :]
                input_R_I_test = self.vali_R[:, col_idx]

                price_R_U_train = self.train_price_R[row_idx, :]
                price_R_I_train = self.train_price_R[:, col_idx]
                price_R_U_test = self.vali_price_R[row_idx, :]
                price_R_I_test = self.vali_price_R[:, col_idx]
                topt_1 = time.time()
                _, cost = self.sess.run(  # do the optimization by the minibatch
                    [self.optimizer, self.cost_final],
                    feed_dict={
                        self.input_R_U: input_R_U,
                        self.input_R_I: input_R_I,
                        self.input_OH_I: self.I_OH_mat[col_idx, :],
                        self.input_P_cor: p_input,
                        self.input_N_cor: n_input,
                        self.input_price_R_U: price_R_U_train,
                        self.input_price_R_I: price_R_I_train,
                        self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                        self.col_idx: np.reshape(col_idx, (len(col_idx), 1)),
                        self.max_losses: np.array(self.max_losses_np).reshape(1, len(self.losses_names))})
                wandb_log({'time_opt_1': time.time() - topt_1}, commit=True)
                batch_cost = batch_cost + cost
                print '.',
                topt_3 = time.time()
                cost_test = self.sess.run(  # do the optimization by the minibatch
                    self.cost_final,
                    feed_dict={
                        self.input_R_U: input_R_U_test,
                        self.input_R_I: input_R_I_test,
                        self.input_OH_I: self.I_OH_mat[col_idx, :],
                        self.input_P_cor: p_input_test,
                        self.input_N_cor: n_input_test,
                        self.input_price_R_U: price_R_U_test,
                        self.input_price_R_I: price_R_I_test,
                        self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                        self.col_idx: np.reshape(col_idx, (len(col_idx), 1)),
                        self.max_losses: np.array(self.max_losses_np).reshape(1, len(self.losses_names))})

                wandb_log({'time_opt_3': time.time() - topt_3}, commit=True)
                batch_cost_test = batch_cost_test + cost_test
                print '*',
        print("Done itetrating over train and test batches.")
        wandb_log({'time_optimization': time.time() -
                   t_start__batch_training}, commit=True)

        self.save_model()

        t_start__forward_pass = time.time()
        # Evaluate epoch result
        _, Decoder = self.sess.run([self.cost_final, self.Decoder],
                                   feed_dict={
            self.input_R_U: self.train_R,
            self.input_R_I: self.train_R,
            self.input_OH_I: self.I_OH_mat,
            self.input_P_cor: [[0, 0]],
            self.input_N_cor: [[0, 0]],
            self.input_price_R_U: price_R_U_train,
            self.input_price_R_I: price_R_I_train,
            self.row_idx: np.reshape(xrange(self.num_rows), (self.num_rows, 1)),
            self.col_idx: np.reshape(xrange(self.num_cols), (self.num_cols, 1)),
            self.max_losses: np.array(self.max_losses_np).reshape(1, len(self.losses_names))})
        print 'd',
        wandb_log({'time_forward_pass': time.time() -
                   t_start__forward_pass}, commit=True)
        if self.base == 'i':
            [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(Decoder.T, self.vali_R.T,
                                                                                                 self.train_R.T, self.vali_price_R.T)
        else:
            print("Select only some parts of it.")
            [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(
                Decoder, self.vali_R, self.train_R, self.vali_price_R)
        metrics = {'f1': f_score, 'NDCG': NDCG, 'revenue': revenue}

        wandb_dict = {'loss': cost, 'test_loss': cost_test}
        for i, at_n in enumerate([1, 2, 3, 4, 5, 10, 15]):
            wandb_dict['f1_at' + str(at_n)] = f_score[i]
            wandb_dict['NDCG_at' + str(at_n)] = NDCG[i]
            wandb_dict['revenue_at' + str(at_n)] = revenue[i]
        wandb_log(wandb_dict, commit=True)

        # Saving current metrics to a csv
        # =========================================================
        #
        subpath = self._make_subpath()
        model_path = '/'.join(['trained_models', subpath, wandb.run.name, ''])
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
        csv_columns = ['wandb_group_name', 'wandb_run_name',
                       'epoch'] + list(sorted(wandb_dict.keys()))

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
        wandb_log({'time_pareto_front': 0.0}, commit=True)

        # End pareto front creation
        if itr % self.display_step == 0:
            print("Training //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(batch_cost),
                  "Elapsed time : %d sec //" % (time.time() - start_time), "Sampling time: %d s //" % (ts))
            print("Test Cost //", "Epoch %d //" %
                  itr, " Tocal cost = {:.2f}".format(batch_cost_test))

        return batch_cost, batch_cost_test, None, None, metrics

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        start_time = time.time()
        _, Decoder = self.sess.run([self.cost_final, self.Decoder],
                                   feed_dict={
            self.input_R_U: self.train_R,
            self.input_R_I: self.train_R,
            self.input_price_R_U: price_R_U_train,
            self.input_price_R_I: price_R_I_train,
            self.input_OH_I: self.I_OH_mat,
            self.input_P_cor: [[0, 0]],
            self.input_N_cor: [[0, 0]],
            self.row_idx: np.reshape(xrange(self.num_rows), (self.num_rows, 1)),
            self.col_idx: np.reshape(xrange(self.num_cols), (self.num_cols, 1)),
            self.max_losses: np.array(self.max_losses_np).reshape(1, len(self.losses_names))})
        if itr % self.display_step == 0:

            pre_numerator = np.multiply((Decoder - self.vali_R), self.vali_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))

            if itr % 1 == 0:
                if self.base == 'i':
                    [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(Decoder.T, self.vali_R.T,
                                                                                                         self.train_R.T, self.vali_price_R.T)
                else:
                    [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(
                        Decoder, self.vali_R, self.train_R, self.vali_price_R)

            print(
                "Testing //", "Epoch %d //" % itr, " Total cost = {:.2f}".format(
                    numerator),
                " RMSE = {:.5f}".format(RMSE),
                "Elapsed time : %d sec" % (time.time() - start_time))
            print("=" * 100)

    def make_records(self):  # record all the results' details into files
        print("in make record. we will calculate metrics after here.")
        _, Decoder = self.sess.run([self.cost_final, self.Decoder],
                                   feed_dict={
            self.input_R_U: self.train_R,
            self.input_R_I: self.train_R,
            self.input_price_R_U: price_R_U_train,
            self.input_price_R_I: price_R_I_train,
            self.input_OH_I: self.I_OH_mat,
            self.input_P_cor: [[0, 0]],
            self.input_N_cor: [[0, 0]],
            self.row_idx: np.reshape(xrange(self.num_rows), (self.num_rows, 1)),
            self.col_idx: np.reshape(xrange(self.num_cols), (self.num_cols, 1)),
            self.max_losses: np.array(self.max_losses_np).reshape(1, len(self.losses_names))})
        if self.base == 'i':
            [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(
                Decoder.T, self.vali_R.T, self.train_R.T, self.vali_price_R.T, 'save_intermediate_results')
        else:
            [precision, recall, f_score, NDCG, revenue, all_results_df] = utility.test_model_all(
                Decoder, self.vali_R, self.train_R, self.vali_price_R, 'save_intermediate_results')

        utility.metric_record(precision, recall, f_score,
                              NDCG, self.args, self.metric_path)

        utility.test_model_factor(Decoder, self.vali_R, self.train_R)

        return precision, recall, f_score, NDCG, revenue

    @staticmethod
    def l2_norm(tensor):
        # @Esla Multiobj: changed reduce_sum to reduce_mean
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

    def save_model(self):
        self.saver.save(
            self.sess,
            self.get_model_path() + 'jca-model'
        )

    def get_model_path(self):
        if self.model_path is None:
            subpath = self._make_subpath()
            self.model_path = '/'.join(['trained_models',
                                        subpath, wandb.run.name, ''])

        return self.model_path

    def load_model(self):
        if tf.train.latest_checkpoint(self.get_model_path()) is None:
            self.save_model()

        else:
            self.saver.restore(
                self.sess, tf.train.latest_checkpoint(self.get_model_path()))
