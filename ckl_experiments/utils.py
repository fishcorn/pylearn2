import numpy as np
from jobman.tools import DD
import os.path as op


def read_yaml(yaml_file):
    #assert(yaml_file in ['mnist', 'svhn', 'cifar10', 'cifar100'])
    #pwd = op.dirname(op.realpath(__file__))
    #return open('%s/%s_tmpl.yaml' % (pwd, yaml_file)).read()
    return open(yaml_file, 'r').read()

def uniform(a, b):
    return np.random.uniform(a, b)


def log_uniform(a, b, integral=False):
    """
    Randomly selects numbers uniformly at random in the log domain.

    Parameters
    ----------
    a : float
        Lower bound (inclusive).
    b : float
        Upper bound (exclusive).
    integral : bool
        Whether the result should be an integer.

    Returns
    -------
    result : float
        Random float, or nearest integer if requested, in [log(a), log(b)).
    """
    result = np.exp(np.random.uniform(np.log(a), np.log(b)))

    if integral:
        result = round(result)

    return float(result)


def power_of(a, b, base=2.):
    """
    Randomly selects a number in [base^a, base^b).

    Parameters
    ----------
    a : float
        Lower bound (inclusive) on exponent.

    b : float
        Upper bound (exclusive) on exponent.

    base: float
        The base of the exponentiation.

    Return
    ------
    result : integer
        A value between [base^a, base^b).
    """
    assert(type(base) == float)

    power = np.random.randint(a, b)
    return np.power(base, power)

def choose(*args):
    choice = np.random.randint(len(args))
    return args[choice]

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels

    dd = DD()
    # dd['test_bx0_col_norms_max'] = channels['test_bx0_col_norms_max'].val_record[-1]
    # dd['test_bx0_col_norms_mean'] = channels['test_bx0_col_norms_mean'].val_record[-1]
    # dd['test_bx0_col_norms_min'] = channels['test_bx0_col_norms_min'].val_record[-1]
    # dd['test_bx0_max_x_max_u'] = channels['test_bx0_max_x_max_u'].val_record[-1]
    # dd['test_bx0_max_x_mean_u'] = channels['test_bx0_max_x_mean_u'].val_record[-1]
    # dd['test_bx0_max_x_min_u'] = channels['test_bx0_max_x_min_u'].val_record[-1]
    # dd['test_bx0_mean_x_max_u'] = channels['test_bx0_mean_x_max_u'].val_record[-1]
    # dd['test_bx0_mean_x_mean_u'] = channels['test_bx0_mean_x_mean_u'].val_record[-1]
    # dd['test_bx0_mean_x_min_u'] = channels['test_bx0_mean_x_min_u'].val_record[-1]
    # dd['test_bx0_min_x_max_u'] = channels['test_bx0_min_x_max_u'].val_record[-1]
    # dd['test_bx0_min_x_mean_u'] = channels['test_bx0_min_x_mean_u'].val_record[-1]
    # dd['test_bx0_min_x_min_u'] = channels['test_bx0_min_x_min_u'].val_record[-1]
    # dd['test_bx0_range_x_max_u'] = channels['test_bx0_range_x_max_u'].val_record[-1]
    # dd['test_bx0_range_x_mean_u'] = channels['test_bx0_range_x_mean_u'].val_record[-1]
    # dd['test_bx0_range_x_min_u'] = channels['test_bx0_range_x_min_u'].val_record[-1]
    # dd['test_bx0_row_norms_max'] = channels['test_bx0_row_norms_max'].val_record[-1]
    # dd['test_bx0_row_norms_mean'] = channels['test_bx0_row_norms_mean'].val_record[-1]
    # dd['test_bx0_row_norms_min'] = channels['test_bx0_row_norms_min'].val_record[-1]
    # dd['test_h0_col_norms_max'] = channels['test_h0_col_norms_max'].val_record[-1]
    # dd['test_h0_col_norms_mean'] = channels['test_h0_col_norms_mean'].val_record[-1]
    # dd['test_h0_col_norms_min'] = channels['test_h0_col_norms_min'].val_record[-1]
    # dd['test_objective'] = channels['test_objective'].val_record[-1][0]
    # dd['test_y_misclass'] = channels['test_y_misclass'].val_record[-1]
    # dd['test_y_nll'] = channels['test_y_nll'].val_record[-1]
    # dd['train_bx0_col_norms_max'] = channels['train_bx0_col_norms_max'].val_record[-1]
    # dd['train_bx0_col_norms_mean'] = channels['train_bx0_col_norms_mean'].val_record[-1]
    # dd['train_bx0_col_norms_min'] = channels['train_bx0_col_norms_min'].val_record[-1]
    # dd['train_bx0_max_x_max_u'] = channels['train_bx0_max_x_max_u'].val_record[-1]
    # dd['train_bx0_max_x_mean_u'] = channels['train_bx0_max_x_mean_u'].val_record[-1]
    # dd['train_bx0_max_x_min_u'] = channels['train_bx0_max_x_min_u'].val_record[-1]
    # dd['train_bx0_mean_x_max_u'] = channels['train_bx0_mean_x_max_u'].val_record[-1]
    # dd['train_bx0_mean_x_mean_u'] = channels['train_bx0_mean_x_mean_u'].val_record[-1]
    # dd['train_bx0_mean_x_min_u'] = channels['train_bx0_mean_x_min_u'].val_record[-1]
    # dd['train_bx0_min_x_max_u'] = channels['train_bx0_min_x_max_u'].val_record[-1]
    # dd['train_bx0_min_x_mean_u'] = channels['train_bx0_min_x_mean_u'].val_record[-1]
    # dd['train_bx0_min_x_min_u'] = channels['train_bx0_min_x_min_u'].val_record[-1]
    # dd['train_bx0_range_x_max_u'] = channels['train_bx0_range_x_max_u'].val_record[-1]
    # dd['train_bx0_range_x_mean_u'] = channels['train_bx0_range_x_mean_u'].val_record[-1]
    # dd['train_bx0_range_x_min_u'] = channels['train_bx0_range_x_min_u'].val_record[-1]
    # dd['train_bx0_row_norms_max'] = channels['train_bx0_row_norms_max'].val_record[-1]
    # dd['train_bx0_row_norms_mean'] = channels['train_bx0_row_norms_mean'].val_record[-1]
    # dd['train_bx0_row_norms_min'] = channels['train_bx0_row_norms_min'].val_record[-1]
    # dd['train_h0_col_norms_max'] = channels['train_h0_col_norms_max'].val_record[-1]
    # dd['train_h0_col_norms_mean'] = channels['train_h0_col_norms_mean'].val_record[-1]
    # dd['train_h0_col_norms_min'] = channels['train_h0_col_norms_min'].val_record[-1]
    # dd['train_h0_row_norms_max'] = channels['train_h0_row_norms_max'].val_record[-1]
    # dd['train_h0_row_norms_mean'] = channels['train_h0_row_norms_mean'].val_record[-1]
    # dd['train_h0_row_norms_min'] = channels['train_h0_row_norms_min'].val_record[-1]
    # dd['train_objective'] = channels['train_objective'].val_record[-1][0]
    # dd['train_y_misclass'] = channels['train_y_misclass'].val_record[-1]
    # dd['train_y_nll'] = channels['train_y_nll'].val_record[-1]
    # dd['valid_bx0_col_norms_max'] = channels['valid_bx0_col_norms_max'].val_record[-1]
    # dd['valid_bx0_col_norms_mean'] = channels['valid_bx0_col_norms_mean'].val_record[-1]
    # dd['valid_bx0_col_norms_min'] = channels['valid_bx0_col_norms_min'].val_record[-1]
    # dd['valid_bx0_max_x_max_u'] = channels['valid_bx0_max_x_max_u'].val_record[-1]
    # dd['valid_bx0_max_x_mean_u'] = channels['valid_bx0_max_x_mean_u'].val_record[-1]
    # dd['valid_bx0_max_x_min_u'] = channels['valid_bx0_max_x_min_u'].val_record[-1]
    # dd['valid_bx0_mean_x_max_u'] = channels['valid_bx0_mean_x_max_u'].val_record[-1]
    # dd['valid_bx0_mean_x_mean_u'] = channels['valid_bx0_mean_x_mean_u'].val_record[-1]
    # dd['valid_bx0_mean_x_min_u'] = channels['valid_bx0_mean_x_min_u'].val_record[-1]
    # dd['valid_bx0_min_x_max_u'] = channels['valid_bx0_min_x_max_u'].val_record[-1]
    # dd['valid_bx0_min_x_mean_u'] = channels['valid_bx0_min_x_mean_u'].val_record[-1]
    # dd['valid_bx0_min_x_min_u'] = channels['valid_bx0_min_x_min_u'].val_record[-1]
    # dd['valid_bx0_range_x_max_u'] = channels['valid_bx0_range_x_max_u'].val_record[-1]
    # dd['valid_bx0_range_x_mean_u'] = channels['valid_bx0_range_x_mean_u'].val_record[-1]
    # dd['valid_bx0_range_x_min_u'] = channels['valid_bx0_range_x_min_u'].val_record[-1]
    # dd['valid_bx0_row_norms_max'] = channels['valid_bx0_row_norms_max'].val_record[-1]
    # dd['valid_bx0_row_norms_mean'] = channels['valid_bx0_row_norms_mean'].val_record[-1]
    # dd['valid_bx0_row_norms_min'] = channels['valid_bx0_row_norms_min'].val_record[-1]
    # dd['valid_h0_col_norms_max'] = channels['valid_h0_col_norms_max'].val_record[-1]
    # dd['valid_h0_col_norms_mean'] = channels['valid_h0_col_norms_mean'].val_record[-1]
    # dd['valid_h0_col_norms_min'] = channels['valid_h0_col_norms_min'].val_record[-1][0]
    # dd['valid_objective'] = channels['valid_objective'].val_record[-1]
    # dd['valid_y_misclass'] = channels['valid_y_misclass'].val_record[-1]
    # dd['valid_y_nll'] = channels['valid_y_nll'].val_record[-1]

    return dd

    #train_y_misclass = channels['y_misclass'].val_record[-1]
    #return DD(train_y_misclass=train_y_misclass)
