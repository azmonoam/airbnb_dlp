import pandas as pd
import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
from src.models import create_model
from src.utils.utils import create_dataloader, validate


# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
parser.add_argument('--album_path', type=str, default='./albums/Graduation/0_92024390@N00')
parser.add_argument('--val_dir', type=str, default='./albums') #  /Graduation') # /0_92024390@N00')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--remove_model_jit', type=int, default=None)


#def get_album(args):

#    files = os.listdir(args.album_path)
#    n_files = len(files)
#    idx_fetch = np.linspace(0, n_files-1, args.album_clip_length, dtype=int)
#    tensor_batch = torch.zeros(len(idx_fetch), args.input_size, args.input_size, 3)
#    for i, id in enumerate(idx_fetch):
#        im = Image.open(os.path.join(args.album_path, files[id]))
#        im_resize = im.resize((args.input_size, args.input_size))
#        np_img = np.array(im_resize, dtype=np.uint8)
#        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
#    tensor_batch = tensor_batch.permute(0, 3, 1, 2)#.cuda()   # HWC to CHW
    # tensor_images = torch.unsqueeze(tensor_images, 0).cuda()
#    montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
#    return tensor_batch, montage


#def inference(tensor_batch, model, classes_list, args):

#    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
#    np_output = output.cpu().detach().numpy()
#    idx_sort = np.argsort(-np_output)
    # Top-k
#    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
#    scores = np_output[idx_sort][: args.top_k]
    # Threshold
#    idx_th = scores > args.threshold
#    return detected_classes[idx_th], scores[idx_th]


#def display_image(im, tags, filename, path_dest):

#    if not os.path.exists(path_dest):
#        os.makedirs(path_dest)

#    plt.figure()
#    plt.imshow(im)
#    plt.axis('off')
#    plt.axis('tight')
#    plt.rcParams["axes.titlesize"] = 16
#    plt.title("Predicted classes: {}".format(tags))
#    print(os.path.join(path_dest, filename))
#    plt.savefig(os.path.join(path_dest, filename))


def main(ids):

    args = parser.parse_args()

    prefix = args.path_output

    meaningful_images = pd.DataFrame(columns = ['ids', 'meaningful_im','is_first'])

    for id in ids:
        album_path = '{}{}_attn.pt'.format(prefix, id)
        order_path = '{}{}_order.pt'.format(prefix, id)
        album_attention = torch.load(album_path)
        album_order = torch.load(order_path)
        arg_max = int(torch.argmax(album_attention[0, 1:]))
        meaningful_image = album_order[arg_max]
        is_first = (meaningful_image[-1] == '0')
        meaningful_images = pd.concat([meaningful_images, pd.DataFrame({"ids":[id], 'meaningful_im':[meaningful_image], "is_first":[is_first]})])


    guessed_baseline = meaningful_images.is_first.sum()
    guessed_baseline = guessed_baseline/len(ids)

    return meaningful_images, guessed_baseline
    print('Done\n')


if __name__ == '__main__':
    ids = pd.read_csv('/Users/talsokolov/Desktop/airbnb_dlp/airbnb_reg/all_ids.csv')['id'].values
    main(ids)