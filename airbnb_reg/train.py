import pandas as pd
import torch
import time
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
from src.models import create_model
from src.utils.utils import create_dataloader, validate
from src.models.aggregate.layers.transformer_aggregate import TAggregate
from src.loss_functions.asymmetric_loss import AsymmetricLoss
import datetime

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
parser.add_argument('--epochs', type=str, default=1000)
parser.add_argument('--lr', type=str, default=0.001)


def get_album(args):
    files = os.listdir(args.album_path)
    n_files = len(files)
    idx_fetch = np.linspace(0, n_files - 1, args.album_clip_length, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), args.input_size, args.input_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(args.album_path, files[id]))
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2).cuda()   # HWC to CHW
    # tensor_images = torch.unsqueeze(tensor_images, 0).cuda()
    montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
    return tensor_batch, montage


def inference(tensor_batch, model, classes_list, args):
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    # Threshold
    idx_th = scores > args.threshold
    return detected_classes[idx_th], scores[idx_th]


def display_image(im, tags, filename, path_dest):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {}".format(tags))
    print(os.path.join(path_dest, filename))
    plt.savefig(os.path.join(path_dest, filename))


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
    #criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    # visualizer = utils2.Visualizer()
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
        if i % 10 == 0:
            torch.save(model.state_dict(), '{}/wights/{}_model_ep_{}.pth'.format(args.results_path, now_ts, i))
            loss_data.to_csv('{}/losses/looses_{}.csv'.format(args.results_path, now_ts))
            loss_grouped_data = loss_data.groupby(['epoch'], as_index=False).median()
            plt.title('loss as function of epochs')
            plt.plot(loss_grouped_data['epoch'], loss_grouped_data['loss'])
            plt.savefig('{}/losses/looses_{}_ep_{}.jpg'.format(args.results_path, now_ts, str(i-1)))

    # Get album
    tensor_batch, montage = get_album(args)

    # Inference
    tags, confs = inference(tensor_batch, model, classes_list, args)

    # Visualization
    display_image(montage, tags, 'result.jpg', os.path.join(args.path_output, args.album_path).replace("./albums", ""))

    # Actual validation process
    # print('loading album and doing inference...')
    # map = validate(model, val_loader, classes_list, args.threshold)
    # print("final validation map: {:.2f}".format(map))

    print('Done\n')

if __name__ == '__main__':
    main()
