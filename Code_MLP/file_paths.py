import os


def ensure_dir_exists(file_path):
    # Ensure directories are created if necessary
    #
    target_directory = os.path.dirname(file_path)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)


def generate_base_path(base_folder, input_params):
    # Example: '/ds/ML1M_subsampled/1'
    #

    return '/'.join([
        './' + base_folder,
        input_params['dataset_folder_ident'],
        input_params['dataset_number']
    ])


def trained_model_file_name(input_params):
    return _generate_file_name('trained_models', input_params, suffix=input_params['algorithm_key'])


def predictions_file_name(input_params):
    return _generate_file_name('trained_models', input_params, suffix=input_params['algorithm_key']+'_pred')


def generate_dataset_file_name(input_params):
    return _generate_file_name('ds', input_params)


def preprocessed_dataset_file_name(input_params):
    return _generate_file_name('ds', input_params, suffix='_preprocessed')


def reranking_threshold_file_name(input_params):
    return _generate_file_name('plots', input_params, suffix='__reranking_threshold', file_type='.csv')


def prediction_histogram_file_name(input_params):
    return _generate_file_name('plots', input_params, suffix='__prediction_histogram', file_type='.csv')


def weighted_revenue_sensitivity_file_name(input_params):
    return _generate_file_name('plots', input_params, suffix='__weighted_revenue_sensitivity', file_type='.csv')


def plot_customer_product_histogram_file_name(input_params):
    return _generate_file_name('plots', input_params, suffix='__product_histogram', file_type='.png')


def plot_price_variance_file_name(input_params):
    return _generate_file_name('plots', input_params, suffix='__price_distribution', file_type='.png')


def _dataset_type(input_params):
    file_name_tokens = []

    # Handling of different RemovalProb / ProductCount
    # Example: ['rm0.7'] OR ['p5']
    #
    ds_type = input_params['dataset_type']
    if ds_type == 'mixed_product_cnt':
        file_name_tokens.append('rm' + str(input_params['removal_prob']))
    elif ds_type == 'separate_product_cnt':
        file_name_tokens.append('p' + str(input_params['target_cnt_product']))
    else:
        raise ValueError("Invalid :dataset_type found: '{}'".format(ds_type))

    return file_name_tokens


def _dataset_order(input_params):
    file_name_tokens = []

    # Handling of different Orders
    # Example ['p5', 'random']
    #
    ds_order = input_params['dataset_order']
    if ds_order == 'random':
        file_name_tokens.append('random')
    elif ds_order == 'temporal':
        ds_order_temp_num = input_params['dataset_order_temporal_num_days']
        if ds_order_temp_num == 0:
            file_name_tokens.append('temporal')
        elif ds_order_temp_num > 0:
            raise NotImplementedError(
                "TODO: Implement :generate_and_store_dataset__filename method using the last :dataset_order_temporal_num_days days.")
        else:
            raise ValueError(
                "Invalid :dataset_order_temporal_num_days found: '{}'".format(ds_order_temp_num))
    else:
        raise ValueError("Invalid :dataset_order found: '{}'".format(ds_order))

    if input_params['train_unordered']:
        file_name_tokens.append('train_unordered')

    return file_name_tokens

# Generates the Pickle file name of the prepared dataset
#


def _generate_file_name(base_folder, input_params, suffix='', file_type='.pkl'):
    base_path = generate_base_path(base_folder, input_params)
    file_name_tokens = []
    file_name_tokens += _dataset_type(input_params)
    file_name_tokens += _dataset_order(input_params)

    if input_params['dataset_type'] == 'mixed_product_cnt':
        file_name_tokens.append('mixed_product')
    file_name_tokens.append('splitted')

    if len(suffix) > 0:
        file_name_tokens.append(suffix)

    file_name_tokens[-1] += file_type
    file_name = '_'.join(file_name_tokens)

    # Example:     './ds/ML1M_subsampled/1/rm0.7_temporal_mixed_product_splitted.pkl'
    #
    return base_path + '/' + file_name
