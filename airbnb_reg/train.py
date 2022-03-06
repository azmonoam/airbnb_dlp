import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random

from src.models import create_model
from src.utils.utils import create_dataloader

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album listings price prediction using Transformers Attention.')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
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
parser.add_argument('--path_output', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/outputs/')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--results_path', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/results')
parser.add_argument('--train_ids_path', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/train_ids.txt')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_rate', type=int, default=10)
parser.add_argument('--save_attention', type=bool, default=True)
parser.add_argument('--save_embeddings', type=bool, default=False)
parser.add_argument('--save_files', type=bool, default=False)
parser.add_argument('--start_ts', type=str, default=datetime.datetime.now().strftime('%d-%m-%y_%H-%M'))


def save_epochs_loss_results(epoch, train_loss_data, test_loss_data, args):
    train_loss_data.to_csv('{}/losses/train_losses_{}.csv'.format(args.results_path, args.start_ts))
    test_loss_data.to_csv('{}/losses/test_losses_{}.csv'.format(args.results_path, args.start_ts))
    train_loss_grouped_data = train_loss_data.groupby(['epoch'], as_index=False).median()
    test_loss_grouped_data = test_loss_data.groupby(['epoch'], as_index=False).median()
    plt.figure()
    plt.title('loss as function of epochs - lr:{}'.format(args.lr))
    plt.plot(train_loss_grouped_data['epoch'], train_loss_grouped_data['loss'])
    plt.plot(test_loss_grouped_data['epoch'], test_loss_grouped_data['loss'])
    plt.legend(['train', 'test'])
    plt.savefig('{}/losses/losses_{}_ep_{}.jpg'.format(args.results_path, args.start_ts, str(epoch)))


def create_album_list(train_val_loader):
    all_album_list = []
    listings = []
    for a, p in train_val_loader:
        all_album_list.append([a, p])
    num_apt = len(all_album_list[0][0])
    batch = 0
    for j in range(0, len(train_val_loader.dataset.samples), num_apt):
        id_list = []
        for i in range(0, len(all_album_list[batch][0])):
            id_list.append(train_val_loader.dataset.samples[j+i][0])
        listings.append(id_list)
        batch += 1
    for i in range(len(all_album_list)):
        all_album_list[i].append(listings[i])
    return all_album_list


def main():
    print('Photo album listings price prediction using Transformers Attention')

    # ----------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()

    # Setup model
    print('creating and loading the model...')
    # state = torch.load(args.model_path, map_location='cpu')
    model = create_model(args).cuda()
    # model.load_state_dict(state['model'], strict=True)
    model.eval()

    # Setup data loader
    print('creating data loaders...')
    train_val_loader = create_dataloader(args, train_mode=True)
    all_album_list = create_album_list(train_val_loader)
    test_val_loader = create_dataloader(args, train_mode=False)
    #test_album_list = create_album_list(test_val_loader)
    print('done\n')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    epochs = args.epochs
    print('start learning')

    train_loss_data = pd.DataFrame(columns=['epoch', 'batch', 'loss'])
    test_loss_data = pd.DataFrame(columns=['epoch', 'batch', 'loss'])
    pred_data = pd.DataFrame(columns=['type', 'id', 'price', 'pred'])

    for i in range(1, epochs+1):
        random.seed(datetime.datetime.now().timestamp())
        random.shuffle(all_album_list)
        batch = 0
        for album_batch, price_batch, images_paths in all_album_list:
            epoch_num = int(i)
            album_batch.requires_grad_()
            album_batch = album_batch.cuda()
            pred = model(album_batch, images_paths, epoch_num)
            pred = pred.to(torch.float)
            price_batch = price_batch.to(torch.float).cuda()
            loss = criterion(pred, price_batch)
            loss = loss.to(torch.float).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            print('train: epoch: {},batch: {}, loss: {}'.format(i, batch, train_loss))
            train_loss_data = pd.concat([train_loss_data, pd.DataFrame({'epoch': [i], 'batch': [batch], 'loss': [train_loss]})],
                                        ignore_index=True, axis=0)
            if i == epochs:
                for j in range(0, pred.shape[0]):
                    ind = j * args.album_clip_length
                    pred_data = pd.concat([pred_data, pd.DataFrame({'type': 'train', 'id':
                        [images_paths[ind][images_paths[ind].find('A') + 1: images_paths[ind].find('_')]], 'price': price_batch[j].detach().cpu(),
                                                                    'pred': pred[j].detach().cpu()})], ignore_index=True,
                                          axis=0)
            batch += 1

        batch = 0
        j = 0
        for album_batch, price_batch in test_val_loader:
            album_batch = album_batch.cuda()
            ## get list of ids ##
            id_list = []
            batch_size = album_batch.shape[0]
            for i in range(0, batch_size):
                id_list.append(test_val_loader.dataset.samples[j + i][0])
            j += batch_size
            ## ## ## ## ## ## ##
            pred = model(album_batch, id_list, epoch_num)
            pred = pred.to(torch.float)
            price_batch = price_batch.to(torch.float).cuda()

            test_loss = criterion(pred, price_batch)
            test_loss = test_loss.to(torch.float).cuda()
            test_loss = test_loss.item()
            print('test: epoch: {},batch: {}, loss: {}'.format(i, batch, test_loss))
            test_loss_data = pd.concat([test_loss_data, pd.DataFrame({'epoch': [i], 'batch': [batch], 'loss': [test_loss]})],
                                        ignore_index=True, axis=0)
            if i == epochs:
                for j in range(0, pred.shape[0]):
                    ind = j * args.album_clip_length
                    pred_data = pd.concat([pred_data, pd.DataFrame({'type': 'test', 'id':
                        [images_paths[ind][images_paths[ind].find('A') + 1: images_paths[ind].find('_')]], 'price': price_batch[j].detach().cpu(),
                                                                    'pred': pred[j].detach().cpu()})], ignore_index=True,
                                          axis=0)
            batch += 1

        if i % args.save_rate == 0:
            torch.save(model.state_dict(), '{}/wights/{}_model_ep_{}.pkl'.format(args.results_path, args.start_ts, i))
            save_epochs_loss_results(i, train_loss_data, test_loss_data, args)
    pred_data.to_csv('{}/losses/predictions_{}.csv'.format(args.results_path, args.start_ts))
    print('Done\n')

if __name__ == '__main__':
    main()
