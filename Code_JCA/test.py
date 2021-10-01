"""
Original code from Ziwei Zhu zhuziwei@tamu.edu
Modified by Yasamin Klingler and Claude lehmann esla@zhaw.ch, lehl@zhaw.ch
"""
import wandb
from JCA import JCA
import os
import argparse
from datetime import datetime
import time
from data_preprocessor import *
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == '__main__':
    # os.system('nvidia-smi')
    print('')
    print('================================================================================================')
    print('CHECK IF THE GPU HAS AVAILABLE MEMORY, IF OK PRESS CTRL+D, OTHERWISE ABORT CTRL+C')
    print('================================================================================================')
    print('')
    import code
    code.interact(local=dict(globals(), **locals()))

    neg_sample_rate = 1

    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())
    data_name = ''
    base = 'u'

    parser = argparse.ArgumentParser(description='JCA')

    parser.add_argument('--method', default='JCA')
    parser.add_argument('--train_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lambda_value', type=float,
                        choices=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001], default=0.001)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--optimizer_method', choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent',
                                                       'Momentum'], default='Adam')
    parser.add_argument(
        '--g_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument(
        '--f_act', choices=['Sigmoid', 'Relu', 'Elu', 'Tanh', "Identity"], default='Sigmoid')
    parser.add_argument('--U_hidden_neuron', type=int, default=160)
    parser.add_argument('--I_hidden_neuron', type=int, default=160)
    parser.add_argument('--base', type=str, default=base)
    parser.add_argument('--neg_sample_rate', type=int, default=neg_sample_rate)

    parser.add_argument('--split_num', default='3')
    parser.add_argument('--losses_names', default='correctness',
                        choices=['revenue', 'correctness', 'revenue_and_correctness'])
    parser.add_argument('--wandb_entity', default='yasies93')
    parser.add_argument('--wandb_project', default='zhaw_nquest')

    # lehl@2021-05-06: This name is used as the identifier for the current experiment
    # and will be checked to see if the experiment was run before. If a checkpoint
    # with that name is found, the run will be continued from the most recent checkpoint
    #
    parser.add_argument('--experiment_identifier', default=None)

    args = parser.parse_args()
    wandb_group_name = 'JCA_' + args.split_num

    if 'losses_names' in args.__dict__.keys():
        if args.losses_names == 'revenue':
            args.losses_names = ['revenue_loss']
            wandb_group_name += '_revenue'

        if args.losses_names == 'correctness':
            args.losses_names = ['correctness_loss']
            wandb_group_name += '_precision'

        if args.losses_names == 'revenue_and_correctness':
            args.losses_names = ['correctness_loss', 'revenue_loss']
            wandb_group_name += '_multiobj'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print('')
    print('================================================================')
    print("Can Tensorflow use the GPU?\t", tf.test.is_gpu_available())
    print('================================================================')
    print('')

    ML1M = ml1m(args.split_num)
    train_R, test_R, train_price_R, test_price_R = ML1M.test()

    metric_path = './metric_results_test/' + date + '/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    metric_path = metric_path + '/' + \
        str(parser.description) + "_" + str(current_time)
    print("entering JCA")

    dataset_dict = dict()
    dataset_dict['train_R'] = train_R.shape
    dataset_dict['test_R'] = test_R.shape
    dataset_dict['train_price_R'] = train_price_R.shape
    dataset_dict['test_price_R'] = test_price_R .shape

    # WANDB Initialization
    #
    WANDB_API_KEY = ''

    # Check for the run name in the arguments or set a default timestamp instead
    #
    if args.experiment_identifier is not None:
        wandb_run_name = args.experiment_identifier
    else:
        # wandb_run_name = str(datetime.now().strftime('%Y-%m-%d__%H%M%S'))
        wandb_run_name = str('cross_validation')

    # Generate the path where the model will be saved
    def _make_subpath(args):
        loss_names = ''
        for loss in args.losses_names:
            loss_names = loss_names + loss + '_'
        # make a directory with model params if not exist.
        subdir_path = 'batch' + str(args.batch_size) + '_' + \
            str(args.method) + '_' + \
            'lr' + str(args.lr) + '_' + \
            'lambda' + str(args.lambda_value) + '_' + \
            'opt' + str(args.optimizer_method) + '_' + \
            'obj' + str(loss_names) + \
            'split' + str(args.split_num)

        return subdir_path
    subpath = _make_subpath(args)

    model_path = '/'.join(['trained_models', subpath,
                           wandb_group_name, wandb_run_name, ''])

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Look if a WandB resume id can be found (saved in a text file). If not,
    # start a completely new run and save the resume id in a text file
    #
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=wandb_group_name,
               name=wandb_run_name, config=dict(args.__dict__, **dataset_dict))
    wandb.log({'current_epoch': 1})

    jca = JCA(sess, args, train_R, test_R, train_price_R,
              test_price_R, metric_path, date, data_name)
    print("JCA object created")
    print("start jca run...")
    jca.run(train_R, test_R, train_price_R, test_price_R)
    print("End.")
