import math

def calc_learning_rate(current_epoch, initial_lr, total_epochs, current_batch, total_batches_per_epoch, lr_schedule_type="cosine"):
    if lr_schedule_type == "cosine":
        # Calculate the total number of iterations (epochs * batches per epoch)
        total_iterations = total_epochs * total_batches_per_epoch
        # Calculate the current iteration
        current_iteration = current_epoch * total_batches_per_epoch + current_batch
        # Calculate the cosine annealing learning rate
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * current_iteration / total_iterations))
        return lr
    elif lr_schedule_type == "":
        # If no schedule type is provided, return the initial learning rate
        return initial_lr
    else:
        # Raise an error for unsupported schedule types
        raise ValueError("do not support: %s" % lr_schedule_type)

