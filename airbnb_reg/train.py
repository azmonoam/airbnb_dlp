import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models import create_model
from src.utils.utils import create_dataloader

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album Event recognition using Transformers Attention.')
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
parser.add_argument('--path_output', type=str, default='./outputs')
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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_rate', type=int, default=10)



def main():
    print('airbnb_reg demo of inference code on a single album.')

    # ----------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()

    # Setup model
    print('creating and loading the model...')
    # state = torch.load(args.model_path, map_location='cpu')
    # args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    # model.load_state_dict(state['model'], strict=True)
    model.eval()
    idx_to_class = {i: i for i in range(16)}
    idx_to_class.update({i: i+1 for i in range(16, 20)})
    classes_list = np.array(list(idx_to_class.values()))

    # Setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    epochs = args.epochs
    print('start learning')

    losses = []
    loss_data = pd.DataFrame(columns=['epoch', 'batch', 'loss'])
    now_ts = datetime.datetime.now().strftime('%d-%m-%y %H:%M')
    for i in range(epochs):
        batch = 0
        for album_batch, price_batch in val_loader:
            album_batch.requires_grad_()
            album_batch = album_batch.cuda()
            pred = model(album_batch)
            pred = pred.to(torch.float)
            price_batch = price_batch.to(torch.float).cuda()
            loss = criterion(pred, price_batch)
            loss = loss.to(torch.float).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_number = loss.item()
            losses.append((loss_number, i, batch))
            print('epoch: {},batch: {}, loss: {}'.format(i, batch, loss_number))
            loss_data = loss_data.append({'epoch': i, 'batch': batch, 'loss': loss_number}, ignore_index=True)
            batch += 1
        if i % args.save_rate == 0:
            torch.save(model.state_dict(), '{}/wights/{}_model_ep_{}.pkl'.format(args.results_path, now_ts, i))
            loss_data.to_csv('{}/losses/looses_{}.csv'.format(args.results_path, now_ts))
            loss_grouped_data = loss_data.groupby(['epoch'], as_index=False).median()
            plt.title('loss as function of epochs')
            plt.plot(loss_grouped_data['epoch'], loss_grouped_data['loss'])
            plt.savefig('{}/losses/looses_{}_ep_{}.jpg'.format(args.results_path, now_ts, str(i-1)))


    print('Done\n')

if __name__ == '__main__':
    main()
