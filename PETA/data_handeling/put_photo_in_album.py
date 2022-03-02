import os
import pandas as pd
import shutil

PATH = '/Users/leeatgen/PycharmProjects/PETA/albums/airbnb'
PHOTO_PATH = '/Users/leeatgen/PycharmProjects/PETA/albums/airbnb_combined'
CSV_PATH = '/Users/leeatgen/PycharmProjects/PETA/all_filtered.csv'

id_df = pd.read_csv(CSV_PATH)
id_list = id_df['id'].to_list()
for id in id_list:
    cost = id_df['scaled_price'][id_df['id'] == id].item()
    if str(cost) not in os.listdir(PATH):
        os.mkdir('{}/{}'.format(PATH, str(cost)))
    new_dir = '{}/{}/{}'.format(PATH, str(cost), str(id))
    os.mkdir(new_dir)
    num_foud = 0
    im_idx = 0
    while num_foud < 5:
        photo_name_jp = "A{}_I{}.jpeg".format(id, im_idx)
        photo_name_pn = "A{}_I{}.png".format(id, im_idx)
        if photo_name_jp in os.listdir(PHOTO_PATH):
            shutil.copy( "{}/{}".format(PHOTO_PATH, photo_name_jp), "{}/{}".format(new_dir, photo_name_jp))
            num_foud += 1
        elif photo_name_pn in os.listdir(PHOTO_PATH):
            shutil.copy( "{}/{}".format(PHOTO_PATH, photo_name_pn), "{}/{}".format(new_dir, photo_name_pn))
            num_foud += 1
        im_idx+=1

