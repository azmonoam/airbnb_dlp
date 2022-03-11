from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from src.utils.utils import create_dataloader
import argparse
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm

parser = argparse.ArgumentParser(
    description='airbnb_reg: Photo album listings price prediction using Transformers Attention.')

parser.add_argument('--album_path', type=str, default='/home/labs/testing/class63/airbnb')
parser.add_argument('--val_dir', type=str, default='/home/labs/testing/class63/airbnb')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='/home/labs/testing/class63/airbnb')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--album_clip_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--results_path', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/results')
parser.add_argument('--train_ids_path', type=str, default='/home/labs/testing/class63/airbnb_reg/train_ids.txt')



def main():
    args = parser.parse_args()
    lr = LinearRegression()
    train_val_loader = create_dataloader(args, train_mode=True)
    test_val_loader = create_dataloader(args, train_mode=False)
    resnet152_model = torchvision.models.resnet152(pretrained=True)
    model = nn.Sequential(*(list(resnet152_model.children())[:-1]))
    train_x, train_y = [], []
    print("getting embedding from ResNet152 on train set...")
    for input, target in train_val_loader:
        embedding = model(input)
        embedding_per_apt = embedding.tensor_split(int(input.shape[0] / 5))
        train_x.extend(
            [embedding_per_apt[i].squeeze().flatten().detach().tolist() for i in range(len(embedding_per_apt))])
        train_y.extend(target.squeeze().tolist())

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    print("ready! fitting linear regression model ...")
    lr.fit(train_x, train_y)

    test_x, test_y = [], []
    print("getting embeddings from ResNet152 on test set... ")
    for input, target in test_val_loader:
        embedding = model(input)
        embedding_per_apt = embedding.tensor_split(int(input.shape[0] / 5))
        test_x.extend(
            [embedding_per_apt[i].squeeze().flatten().detach().tolist() for i in range(len(embedding_per_apt))])
        try:
            test_y.extend(target.squeeze().tolist())
        except TypeError:
            test_y.extend([target.squeeze().tolist()])

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    print("computing prediction on test set embeddings...")
    pred_y = lr.predict(test_x)

    mse = metrics.mean_squared_error(test_y, pred_y)
    print("Mean Squared Error {}".format(mse))


if __name__ == '__main__':
    main()
