import pandas as pd
import torch
import time
import utils2
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

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
parser.add_argument('--album_path', type=str, default='/Users/leeatgen/airbnb_dlp/airbnb_exdata')
parser.add_argument('--val_dir', type=str, default='/Users/leeatgen/airbnb_dlp/airbnb_exdata')  # /Graduation') # /0_92024390@N00')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='/Users/leeatgen/airbnb_dlp/airbnb_exdata')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--train_ids_path', type=str, default='/Users/leeatgen/airbnb_dlp/train_ids.txt')

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
    model = create_model(args)#.cuda()
    # model.load_state_dict(state['model'], strict=True)
    model.eval()
    idx_to_class = {i: i for i in range(16)}
    idx_to_class.update({i: i+1 for i in range(16, 20)})
    classes_list = np.array(list(idx_to_class.values()))
    # print('Class list:', classes_list)

    # Setup data loader
    print('creating data loader...')
    train_val_loader = create_dataloader(args, train_mode=True)
    test_val_loader = create_dataloader(args, train_mode=False)
    print('done\n')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = torch.nn.CrossEntropyLoss()
    epochs = 1
    # visualizer = utils2.Visualizer()
    print('start learning')

    losses = []
    for i in range(epochs):
        #t0 = time.time()
        batch = 0
        for album_batch, price_batch in train_val_loader:
            album_batch.requires_grad_()
            album_batch = album_batch#.cuda()
            t1 = time.time()
            pred = model(album_batch)
            t2 = time.time()
            print("model time {}".format(str(t2 - t1)))
            pred = pred.to(torch.float)
            #if i%10 == 0:
            #    pred_table = pd.DataFrame()
            #    pred_table['ids'] = ids
            #    pred_table['pred'] = pred
            #    pred_table.to_csv('pred_table_epoch{}.csv'.format(i), index=False)
            price_batch = price_batch.to(torch.float)#.cuda()
            # print(album, pred, price)
            loss = criterion(pred, price_batch)
            loss = loss.to(torch.float)#.cuda()
            # acc = accuracy(pred, price)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #t3 = time.time()
            #print("batch time {}".format(str(t3 - t0)))

            loss_number = loss.item()
            losses.append((loss_number, i, batch))
            # visualizer.update(loss_number)
            print('epoch: {},batch: {}, loss: {}'.format(i, batch, loss_number))
            batch += 1
        if i % 10 == 0:
            torch.save(model.state_dict(), f'/Users/leeatgen/airbnb_dlp/model_saving/model_{i}.pkl')

        for album_batch, price_batch in test_val_loader:
            album_batch = album_batch#.cuda()
            #t1 = time.time()
            pred = model(album_batch)
            #t2 = time.time()
            #print("model time {}".format(str(t2 - t1)))
            pred = pred.to(torch.float)
            #if i%10 == 0:
            #    pred_table = pd.DataFrame()
            #    pred_table['ids'] = ids
            #    pred_table['pred'] = pred
            #    pred_table.to_csv('test_pred_table_epoch{}.csv'.format(i), index=False)
            price_batch = price_batch.to(torch.float)#.cuda()
            # print(album, pred, price)
            loss = criterion(pred, price_batch)
            loss = loss.to(torch.float)#.cuda()

        ## TODO : load test data loader with val_loader = create_dataloader(args, train=False) and run test epoch


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


from utils2 import Metric, accuracy

__all__ = ['test_epoch', 'test_epoch', 'train_loop']


#################################################
# train_epoch
#################################################

def train_epoch(model, criterion, optimizer, loader, device):
    """Trains over an epoch, and returns the accuracy and loss over the epoch.

  Note: The accuracy and loss are average over the epoch. That's different from
  running the classifier over the data again at the end of the epoch, as the
  weights changed over the iterations. However, it's a common practice, since
  iterating over the training set (again) is time and resource exhustive.

  Note: You MUST have `loss` tensor with the loss value, and `acc` tensor with
  the accuracy (you can use the imported `accuracy` method).

  Args:
    model (torch.nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (torch.optim.Optimizer): The optimizer.
    loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
    loss_metric = Metric()
    acc_metric = Metric()
    for x, y in loader:
        x, y = x.to(device=device), y.to(device=device)
        # BEGIN SOLUTION
        # raise NotImplementedError
        x = x.squeeze()
        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        pred = model(x)
        loss = criterion(pred, y)
        acc = accuracy(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # END SOLUTION
        loss_metric.update(loss.item(), x.size(0))
        acc_metric.update(acc.item(), x.size(0))
    return loss_metric, acc_metric


#################################################
# test_epoch
#################################################

def test_epoch(model, criterion, loader, device):
    """Evaluating the model at the end of the epoch.

  Note: You MUST have `loss` tensor with the loss value, and `acc` tensor with
  the accuracy (you can use the imported `accuracy` method).

  Args:
    model (torch.nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
    loss_metric = Metric()
    acc_metric = Metric()
    for x, y in loader:
        x, y = x.to(device=device), y.to(device=device)
        # BEGIN SOLUTION
        # raise NotImplementedError
        x = x.squeeze()
        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        pred = model(x)
        loss = criterion(pred, y)
        acc = accuracy(pred, y)
        # END SOLUTION
        loss_metric.update(loss.item(), x.size(0))
        acc_metric.update(acc.item(), x.size(0))
    return loss_metric, acc_metric


#################################################
# PROVIDED: train_loop
#################################################

def train_loop(model, criterion, optimizer, train_loader, test_loader, device, epochs, test_every=1):
    """Trains a model to minimize some loss function and reports the progress.

  Args:
    model (torch.nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (torch.optim.Optimizer): The optimizer.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    test_loader (torch.utils.data.DataLoader): The test set data loader.
    device (torch.device): The device to run on.
    epochs (int): Number of training epochs.
    test_every (int): How frequently to report progress on test data.
  """
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, criterion, optimizer, train_loader, device)
        print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
              f'Loss: {train_loss.avg:7.4g}',
              f'Accuracy: {train_acc.avg:.3f}',
              sep='   ')
        if epoch % test_every == 0:
            test_loss, test_acc = test_epoch(model, criterion, test_loader, device)
            print(' Test', f'Epoch: {epoch:03d} / {epochs:03d}',
                  f'Loss: {test_loss.avg:7.4g}',
                  f'Accuracy: {test_acc.avg:.3f}',
                  sep='   ')


if __name__ == '__main__':
    main()
