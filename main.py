import os
import argparse
import torch
import torch.cuda
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from load import *
from model import CheXpertModel
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from utils import load_data
from viz import *
from heatmap import HeatmapGenerator

# TODO: Turn these into flags with defaults.
N_CLASSES = 14
BATCH_SIZE = 32
EPOCH_NUM = 5
MODEL_PATH = 'models/DenseNet121_3_.pkl'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    The original classifier layer does not contain the GlobalAveragePooling2D
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = tv.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def train(device):
    # TODO: update the model we use with flags.
    model = DenseNet121(N_CLASSES).to(device=device)

    # TODO: update the training data loader we use with flags.
    train_data_loader = data_loader(
        "CheXpert-v1.0-small/train.csv", BATCH_SIZE)
    valid_data_loader = data_loader(
        "CheXpert-v1.0-small/valid.csv", BATCH_SIZE)

    cheXPert = CheXpertModel(
        device, model, train_data_loader, valid_data_loader, EPOCH_NUM)
    # Train the model.
    cheXPert.train()

    return cheXPert


def load_model(model_path, device):
    model = DenseNet121(N_CLASSES).to(device=device)

    # TODO: update the training data loader we use with flags.
    train_data_loader = None
    valid_data_loader = data_loader(
        "CheXpert-v1.0-small/valid.csv", BATCH_SIZE)

    cheXPert = CheXpertModel(
        device, model, train_data_loader, valid_data_loader, EPOCH_NUM)
    cheXPert.model.load_state_dict(torch.load(model_path))

    return cheXPert


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('-dc', '--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('-lm', '--load_model', type=str,
                        help='load existing model, must provide model path')
    parser.add_argument('-pl', '--plot_loss', action='store_true',
                        help='load pickle data to plot loss')
    parser.add_argument('-auroc', '--auroc', action='store_true',
                        help='store auroc plots for all classes')
    parser.add_argument('-ds', '--data_stats', action='store_true',
                        help='generate data visualization graphs')
    parser.add_argument('-hm', '--heat_maps', nargs='+', type=str,
                        help='generate heat_maps of the input image id. Separated by common')

    args = parser.parse_args()
    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.load_model:
        print('Loading model')
        model = load_model(args.load_model, device)
    else:
        print('Training model from scratch')
        model = train(device)

    if args.plot_loss:
        data = load_data(str(EPOCH_NUM) + '.p')
        train_losses = data['train_losses']
        valid_losses = data['valid_losses']
        loss_plot(train_losses, valid_losses)

    # Test the model.
    if args.auroc:
        labels, predictions = model.test()

        class_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                       "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                       "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

        for i, class_name in enumerate(class_names):
            fpr, tpr, thresholds = metrics.roc_curve(
                labels.cpu()[:, i], predictions.cpu()[:, i])
            auc = metrics.auc(fpr, tpr)

            print("%s: AUC = %0.3f" % (class_name, auc))
            plt.title("%s: AUC = %0.3f" % (class_name, auc))
            plt.plot(fpr, tpr, label="%s: AUC = %0.3f" % (class_name, auc))
            plt.xlabel("Specificity")
            plt.ylabel("Sensitivity")
            plt.savefig("plots/%s_roc.png" % class_name)
            plt.clf()

    if args.data_stats:
        train_dataset = CheXpertTorchDataset(
            "CheXpert-v1.0-small/train.csv", None)
        valid_dataset = CheXpertTorchDataset(
            "CheXpert-v1.0-small/valid.csv", None)

        class_names = ["No Finding", "Enlarged Cardiom.", "Cardiomegaly", "Lung Opacity",
                       "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                       "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

        label_stats(train_dataset.labels, class_names, 'Train Label Stats')
        label_stats(valid_dataset.labels, class_names, 'Test Label Stats')

    if args.heat_maps:
        hmg = HeatmapGenerator(model.model, device)
        index = 1
        for src_image_path in args.heat_maps:
            patient_id = src_image_path.split('/')[2]
            hmg.generate(src_image_path, "heat_map/%s_1.png" % patient_id)
            index += 1
