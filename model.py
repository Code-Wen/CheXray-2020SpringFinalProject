import argparse
import time
import torch
import torch.cuda
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from load import CheXpertTorchDataset
from utils import AverageMeter, save_data


# def compute_batch_accuracy(output, target):
#     """
#     Computes the accuracy for a batch
#
#     Input
#     -------
#     output: (n_samples, n_labels) ndarray in float
#     target: (n_samples, n_labels) ndarray in 0 or 1
#     """
#     with torch.no_grad():
#
#         batch_size, n_labels = target.size(0)
#         _, pred = output.max(1)
#         correct = pred.eq(target).sum()
#
#         return correct * 100.0 / batch_size


class CheXpertModel():
    def __init__(self, device, model, trainDataLoader, validDataLoader, numEpochs):
        self.device = device
        self.model = model
        self.trainDataLoader = trainDataLoader
        self.validDataLoader = validDataLoader
        self.numEpochs = numEpochs

    # Reference: https://github.com/thtang/CheXNet-with-localization/blob/master/train.py
    # Reference: https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
    def train(self):
        print("Training...")
        optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.9, 0.999))
        criterion = nn.BCELoss()
        best_val_loss = 999999.
        train_losses = []
        valid_losses = []
        best_val_results = None
        for epoch in range(self.numEpochs):
            train_loss = self._train_one_epoch(epoch, optimizer, criterion)
            valid_loss, valid_results = self._eval_one_epoch(epoch, criterion)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            is_best = valid_loss < best_val_loss
            if is_best:
                best_val_loss = valid_loss
                best_val_results = valid_results
                torch.save(self.model.state_dict(), 'DenseNet121_' + str(epoch + 1) + '_best' + '.pkl')

        save_data(train_losses, valid_losses, best_val_loss, best_val_results, prefix=str(self.numEpochs))

    def _train_one_epoch(self, epoch, optimizer, criterion, print_freq=10):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # accuracy = AverageMeter()

        self.model.train()

        # print("Epoch:", epoch)
        # batch_index = 1
        # epoch_loss = 0.
        end = time.time()
        for i, sample in enumerate(self.trainDataLoader):
            data_time.update(time.time() - end)
            # print("  Batch: ", batch_index)
            image = Variable(sample['image'].to(device=self.device))
            label = Variable(sample['labels'].to(device=self.device))

            optimizer.zero_grad()
            outputs = self.model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), label.size(0))
            # accuracy.update(compute_batch_accuracy(outputs, label).item(), label.size(0))

            # batch_index += 1
            # batch_loss = loss.item()
            # print("  Loss: ", batch_loss)
            # epoch_loss += batch_loss

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          epoch, i, len(self.trainDataLoader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

        return losses.avg

    def _eval_one_epoch(self, epoch, criterion, print_freq=2):
        batch_time = AverageMeter()
        losses = AverageMeter()
        # accuracy = AverageMeter()

        results = []

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(self.validDataLoader):
                image = Variable(sample['image'].to(device=self.device))
                label = Variable(sample['labels'].to(device=self.device))
                outputs = self.model(image)
                loss = criterion(outputs, label)
                batch_time.update(time.time() - end)
                end = time.time()

                losses.update(loss.item(), label.size(0))
                # accuracy.update(compute_batch_accuracy(outputs, label).item(), label.size(0))

                y_true = label.detach().to('cpu').numpy().tolist()
                y_pred = outputs.detach().to('cpu').numpy().tolist()
                results.extend(list(zip(y_true, y_pred)))

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                              i, len(self.validDataLoader), batch_time=batch_time, loss=losses,))

        return losses.avg, results

    def test(self):
        print("Testing...")
        self.model.eval()

        if self.device.type == 'cuda':
            preds = torch.FloatTensor().cuda()
            labels = torch.FloatTensor().cuda()
        else:
            labels = torch.FloatTensor()
            preds = torch.FloatTensor()

        with torch.no_grad():
            for i, sample in enumerate(self.validDataLoader):
                sample_labels = sample['labels'].to(device=self.device)

                # print(labels.device, sample_labels.device)
                labels = torch.cat((labels, sample_labels), 0)

                _, channels, height, width = sample['image'].size()
                sample_pred = self.model(sample['image'].view(-1, channels, height, width).to(device=self.device))
                preds = torch.cat((preds, sample_pred), 0)

        # TODO: Compute AUC from labels and predictions.

        return labels, preds
