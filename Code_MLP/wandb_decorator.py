import time
import wandb

# Decorator to time a function and log the
# time it took to WandB
#


def wandb_timing(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()

        wandb_log_key = "time_{}".format(func.__name__)
        wandb_log_value = t2 - t1

        if 'current_epoch' in wandb.run.summary.keys():
            wandb_dict = {'current_epoch': wandb.run.summary['current_epoch']}
        else:
            wandb_dict = {'current_epoch': 1}

        wandb.log(
            dict(wandb_dict, **{wandb_log_key: wandb_log_value}), commit=False)

        return res
    return wrapper

# Decorator to time a function and log the time it took to WandB,
# will also increase the "current_epoch" --> Only call this ONCE per epoch (at the end)
# OR wrap the "per epoch" training method with this decorator
#


def wandb_timing__end_epoch(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()

        if 'current_epoch' not in wandb.run.summary.keys():
            wandb_dict = {'current_epoch': 1}
        else:
            wandb_dict = {'current_epoch': wandb.run.summary['current_epoch']}

        wandb_dict["time_{}".format(func.__name__)] = t2 - t1
        wandb.log(wandb_dict, commit=True)

        wandb.run.summary['current_epoch'] += 1

        return res
    return wrapper

# Can be used to log values to WandB, but includes the current epoch
#


def wandb_log(my_dict, commit=False):
    if 'current_epoch' in wandb.run.summary.keys():
        wandb_dict = {'current_epoch': wandb.run.summary['current_epoch']}
    else:
        wandb_dict = {'current_epoch': 1}

    wandb.log(dict(wandb_dict, **my_dict), commit=commit)
