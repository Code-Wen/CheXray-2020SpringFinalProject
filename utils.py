import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_data(train_losses, valid_losses, best_val_loss, best_val_results, prefix='default'):
    pickle_path = prefix + '.p'

    data = {}
    data['train_losses'] = train_losses
    data['valid_losses'] = valid_losses
    data['best_val_loss'] = best_val_loss
    data['best_val_results'] = best_val_results

    with open(pickle_path, mode='wb') as f:
        pickle.dump(data, f)


def load_data(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)

    return data
