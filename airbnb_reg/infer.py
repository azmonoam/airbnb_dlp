import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import get_apt_city
import numpy as np

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--base_path', type=str, default='/Users/noamazmon/airbnb_dlp')
parser.add_argument('--results_path', type=str, default='/airbnb_reg/results')
parser.add_argument('--path_output', type=str, default='/airbnb_reg/outputs/')
parser.add_argument('--job_id', type=str, default='09-03-22_16-43-22_lr_0.0001_tb_1')
parser.add_argument('--id_city_path', type=str, default='/airbnb_reg/id_and_city.csv')

def count_image_num_anf_plot(args, color):
    att_df = pd.read_csv(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}.csv")
    count_important_pic = att_df.groupby(['most_important_pic'], as_index=False).count()[['most_important_pic', 'id']]
    bigger_then_four = count_important_pic.loc[count_important_pic['most_important_pic'] > 4].sum()
    smaller_then_four = count_important_pic.loc[count_important_pic['most_important_pic'] <= 4]
    bigger_then_four['most_important_pic'] = 'else'
    data = pd.concat([smaller_then_four, pd.DataFrame(bigger_then_four).transpose()], ignore_index=True, axis=0)
    labels = data['most_important_pic'].tolist()
    sizes = data['id'].tolist()
    labels = [('image ' + str(x)).capitalize() if type(x)==int else x.capitalize() for x in labels ]
    explode = (0.1, 0, 0, 0, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90, colors=color, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    ax1.axis('equal')
    ax1.set_title('Most important photo')
    plt.savefig(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}_most_imp_room_room_number.jpg", dpi=300)
    plt.show()


def get_pred_dist(args):
    predictions_path = args.base_path+ args.results_path + f"/losses/predictions_{args.job_id.split('_lr')[0]}.csv"
    predictions = pd.read_csv(predictions_path)
    pred = predictions['pred']
    gt = predictions['price']
    pred.plot(kind='hist')
    plt.show()
    gt.loc[gt<=2].plot(kind='hist')
    plt.show()


def get_loss_per_city(args):
    pred_full_path = args.base_path + args.results_path + f"/losses/predictions_{args.job_id.split('_lr')[0]}.csv"
    city_id_full_path = args.base_path + args.id_city_path
    pred_with_city_df = get_apt_city.add_city_per_id(city_id_full_path, pred_full_path)
    pred_with_city_df = pred_with_city_df.loc[pred_with_city_df['type'] == 'test']
    pred_with_city_df['loss'] = (pred_with_city_df['price'] - pred_with_city_df['pred'])**2
    avg_loss_per_city = pred_with_city_df.groupby(['city'], as_index=False).median()
    return avg_loss_per_city


def from_loss_to_usd(loss, perc_75, perc_25):
    return ((loss ** 0.5) * ( perc_75 - perc_25))


def add_baselines_and_convert_to_usd(city_conv_data, avg_loss_per_city):
    avg_loss_per_city = pd.concat([avg_loss_per_city, pd.DataFrame(
        columns=['loss_in_usd', 'mean_baseline_loss', 'mean_baseline_in_usd', 'median_baseline_loss',
                 'median_baseline_in_usd'])], axis=1)
    for city in city_conv_data.keys():
        city_25_prec =  city_conv_data[city]['0.25']
        city_75_prec =  city_conv_data[city]['0.75']
        avg_loss_per_city['loss_in_usd'][avg_loss_per_city['city']==city] = from_loss_to_usd(avg_loss_per_city['loss'], city_75_prec, city_25_prec)
        avg_loss_per_city['mean_baseline_loss'][avg_loss_per_city['city']==city] = city_conv_data[city]['mean_baseline_loss']
        avg_loss_per_city['mean_baseline_in_usd'][avg_loss_per_city['city']==city] =  from_loss_to_usd(avg_loss_per_city['mean_baseline_loss'], city_75_prec, city_25_prec)
        avg_loss_per_city['median_baseline_loss'][avg_loss_per_city['city']==city] = city_conv_data[city]['median_baseline_loss']
        avg_loss_per_city['median_baseline_in_usd'][avg_loss_per_city['city']==city] = from_loss_to_usd(avg_loss_per_city['median_baseline_loss'], city_75_prec, city_25_prec)
    return avg_loss_per_city


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        height = round(height,1)
        ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  fontsize= 8,
                        textcoords="offset points",
                        ha='center', va='bottom')


def plot_loss_per_city_and_base_line(args,avg_loss_per_city, color):
    width = 0.35
    labels_short = avg_loss_per_city['city']
    labels = []
    citys = {'ber': 'Berlin', 'nyc': 'New York', 'ist': 'Istanbul','tor':'Toronto', 'gre':'Athens'}
    for city in labels_short:
        labels.append(citys[city])
    x = np.arange(avg_loss_per_city.shape[0])
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, avg_loss_per_city['loss_in_usd'], width, label='loss', color = color[0])
    rects2 = ax.bar(x + width / 2, avg_loss_per_city['median_baseline_in_usd'], width, label='baseline', color = color[5])
    ax.set_ylabel('USD $')
    ax.set_title('Loss in USD per city')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    plt.savefig(args.base_path + args.results_path + f"/losses/predictions_{args.job_id.split('_lr')[0]}_loss_per_city.jpg", dpi=300)
    plt.show()


def add_room_type_to_att_data(args):
    att_df = pd.read_csv(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}.csv")
    room_type_df = pd.read_csv(args.base_path + '/room_type.csv', usecols=['Image_path', 'room_type'])
    att_df_with_most_important_room = att_df.join(room_type_df.set_index('Image_path').rename(columns={"room_type": "I_importat_room_type"}), on='most_important_pic_path')
    att_df_with_most_important_room['room_0_path'] = att_df_with_most_important_room.apply(
        lambda row: row['most_important_pic_path'][:row['most_important_pic_path'].find('I')] + 'I0.jpeg', axis=1)
    att_df_rooms = att_df_with_most_important_room.join(room_type_df.set_index('Image_path').rename(columns={"room_type": "I0_room_type"}), on='room_0_path')
    return att_df_rooms


def plot_room_type_for_important_photo_vs_baseline(att_df_rooms, color):
    I0_data = att_df_rooms.groupby(['I0_room_type'], as_index=False).count()[['I0_room_type', 'id']]
    I0_data = I0_data.iloc[I0_data.I0_room_type.str.lower().argsort()]
    labels = I0_data['I0_room_type'].str.lower().tolist()
    labels = [ x.replace('_', ' ') for x in labels]
    I_importat_data = att_df_rooms.groupby(['I_importat_room_type'], as_index=False).count()[['I_importat_room_type', 'id']]
    I_importat_data = I_importat_data.iloc[I_importat_data.I_importat_room_type.str.lower().argsort()]
    width = 0.35
    x = np.arange(I_importat_data.shape[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, I_importat_data['id'], width, label='prediction',  color=color[0])
    rects2 = ax.bar(x + width / 2, I0_data['id'], width, label='baseline',  color=color[5])
    ax.set_ylabel('number of apartments')
    ax.set_title('most important room distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    plt.savefig(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}_most_imp_room_dist.jpg", dpi=300)
    plt.show()


def plot_loss_usd():
    test_losses = pd.read_csv(args.base_path+ args.results_path + f"/losses/test_losses_{args.job_id}.csv")
    test_losses_loss = test_losses.apply(lambda row: from_loss_to_usd(row['loss'], 142, 51), axis=1)
    test_losses['loss'] = test_losses_loss
    test_medians = test_losses.groupby(['epoch'], as_index=False).median()['loss']

    train_losses = pd.read_csv(args.base_path+ args.results_path + f"/losses/train_losses_{args.job_id}.csv")
    train_losses_loss = train_losses.apply(lambda row: from_loss_to_usd(row['loss'], 142, 51), axis=1)
    train_losses['loss'] = train_losses_loss
    train_medians = train_losses.groupby(['epoch'], as_index=False).median()['loss']

    median_per_epoch = pd.DataFrame({'test': test_medians.values, 'train': train_medians.values, 'median_baseline':from_loss_to_usd(1.136,142,51), 'mean_baseline':from_loss_to_usd(1.0506,142,51)})
    median_per_epoch['epoch'] = [i for i in range(1, len(test_medians) + 1)]

    median_per_epoch.plot('epoch', ['test', 'train', 'median_baseline', 'mean_baseline'], ylabel='loss in USD')
    plt.figure()
    plt.title('loss in USD as function of epochs')
    plt.savefig(args.base_path+ args.results_path + f"/losses/loss_in_usd_over_ep_{args.job_id}.jpg", dpi=300)
    plt.show()


def plot_loss():
    test_losses = pd.read_csv(args.base_path+ args.results_path + f"/losses/test_losses_{args.job_id}.csv")
    test_medians = test_losses.groupby(['epoch'], as_index=False).median()['loss']

    train_losses = pd.read_csv(args.base_path+ args.results_path + f"/losses/train_losses_{args.job_id}.csv")
    train_medians = train_losses.groupby(['epoch'], as_index=False).median()['loss']

    median_per_epoch = pd.DataFrame({'test': test_medians.values, 'train': train_medians.values, 'baseline':1.1364, 'mean_baseline':1.0506})
    median_per_epoch['epoch'] = [i for i in range(1, len(test_medians) + 1)]
    median_per_epoch.plot('epoch', ['test', 'train', 'baseline'], ylabel='loss', color=[color[0],color[4],color[5]])
    plt.title('loss as function of epochs')
    plt.savefig(args.base_path+ args.results_path + f"/losses/loss_over_ep_{args.job_id}.jpg", dpi=300)
    plt.show()


def get_top_and_bottom_apt(args, num_photos):
    predictions_path = args.base_path + args.results_path + f"/losses/predictions_{args.job_id.split('_lr')[0]}.csv"
    predictions = pd.read_csv(predictions_path)
    sorted_test_pred = predictions.loc[predictions['type'] == 'test'].sort_values(by='pred')
    bottom = sorted_test_pred .iloc[:num_photos][['id', 'pred']].reset_index()
    top = sorted_test_pred .iloc[-num_photos:][['id', 'pred']].reset_index().sort_values(by='pred', ascending=False)
    return top, bottom


def get_radical_apt_most_imp_photo(args, apts_df):
    att_df = pd.read_csv(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}.csv",
                         usecols=['id', 'most_important_pic_path'])
    paths = []
    for apt in apts_df['id'].to_list():
        paths.append(att_df[att_df['id']==apt]['most_important_pic_path'].item())
    return paths


def plot_radical_apt_most_imp_photo(args, paths, num_photos, kind):
    num_photos = int(np.sqrt(num_photos))
    f, axarr = plt.subplots(num_photos, num_photos)
    k = 0
    for i in range(num_photos):
        for j in range(num_photos):
            path = args.base_path +'/monk_v1/airbnb/' + paths[k]
            img = mpimg.imread(path)
            axarr[i, j].imshow(img)
            axarr[i, j].axis('off')
            k += 1
            plt.show()
    plt.suptitle(f'Most important images of apartments with the {kind}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig(args.base_path+ args.results_path + f"/{kind}_{args.job_id}.jpg", dpi=300)


def get_highest_att(args, num_apt):
    att_data = pd.read_csv(args.base_path + f"/airbnb_reg/outputs/att_data_{args.job_id.split('_lr')[0]}.csv")
    attֹ_fl_list = []
    most_imp_att_list = []
    for i in range(att_data.shape[0]):
        att = att_data['att_mat'].to_list()[i]
        att_val_list = att[att.find(','):att.find(']')].split(', ')
        attֹ_fl = [float(x) for x in att_val_list if x != '']
        diff_between_tow_most_imp_photos = max(attֹ_fl)/np.sum(attֹ_fl)
        attֹ_fl_list.append(attֹ_fl)
        most_imp_att_list.append(diff_between_tow_most_imp_photos)
    att_data['most_important_pic_att'] = most_imp_att_list
    att_data['first_tow_att'] = attֹ_fl_list
    sorted_att_data = att_data.sort_values(by='most_important_pic_att', ascending=False)
    bottom = sorted_att_data.iloc[:num_apt]
    top = sorted_att_data.iloc[-num_apt:].sort_values(by='most_important_pic_att', ascending=False)
    return bottom, top


def seperate_most_imp_and_rest_photos_paths(sorted_att_data):
    to_plot = []
    for i in range(sorted_att_data.shape[0]):
        ord = sorted_att_data.iloc[i]['pic_order']
        rest_most_important_pic_path = [str(x.strip("'")) for x in ord[ord.find('[')+1:ord.find(']')].split(', ')]
        rest_most_important_pic_path.remove(sorted_att_data.iloc[i]['most_important_pic_path'])
        to_plot.append([sorted_att_data.iloc[i]['most_important_pic_path'], rest_most_important_pic_path])
    return to_plot


def plot_att_diffs_photos(paths, num_apt):
    f, axarr = plt.subplots(num_apt, 5)
    k = 0
    for i in range(num_apt):
        path = args.base_path + '/monk_v1/airbnb/' + paths[k][0]
        img = mpimg.imread(path)
        axarr[i, 0].imshow(img)
        axarr[i, 0].axis('off')
        for j in range(1, 5):
            path = args.base_path + '/monk_v1/airbnb/' + paths[k][1][j - 1]
            img = mpimg.imread(path)
            axarr[i, j].imshow(img)
            axarr[i, j].axis('off')
        k += 1
    plt.show()
    #    plt.suptitle(f'Most important images of apartments with the {kind}')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
#    plt.suptitle(f'Most important images of apartments with the {kind}')



def main(args):
    att_data, guessed_baseline = count_image_num_anf_plot(args, color)
    print(f"predicted the first image {guessed_baseline} time")

    get_pred_dist(args)

    print('Done\n')


if __name__ == '__main__':
    color = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
    args = parser.parse_args()
    city_conv_data = {'nyc': {'0.25': 126.431266846361, '0.75': 248.517520215633, 'mean_baseline_loss': 1.5701, 'median_baseline_loss':1.4448 },
                      'ist': {'0.25': 33.9339622641509, '0.75': 100.161725067385,  'mean_baseline_loss': 0.5891, 'median_baseline_loss': 0.5499 },
                      'ber': {'0.25': 79.9245283018868, '0.75': 156.0,  'mean_baseline_loss':0.9418, 'median_baseline_loss':0.8731 },
                      'tor': {'0.25': 55.3793800539084, '0.75': 143.995956873315,  'mean_baseline_loss': 0.9461, 'median_baseline_loss':0.8662},
                      'gre': {'0.25': 44.2681940700809, '0.75': 89.0,  'mean_baseline_loss': 1.4517, 'median_baseline_loss':1.3287}}
    bottom, top = get_highest_att(args, 3)
    top_paths = seperate_most_imp_and_rest_photos_paths(top)
    bottom_paths = seperate_most_imp_and_rest_photos_paths(bottom)
    plot_att_diffs_photos(top_paths, 3)
    plot_att_diffs_photos(bottom_paths, 3)
    print('plot loss')
    plot_loss()
    print('calc and plot loss per city')
    avg_loss_per_city = get_loss_per_city(args)
    avg_loss_per_city = add_baselines_and_convert_to_usd(city_conv_data, avg_loss_per_city)
    plot_loss_per_city_and_base_line(args, avg_loss_per_city, color)
    print('calc and plot room type distribution')
    att_df_rooms = add_room_type_to_att_data(args)
    plot_room_type_for_important_photo_vs_baseline(att_df_rooms, color)
    print('plot room number pai distribution')
    count_image_num_anf_plot(args, color)
    print('plot most imprtamt pictures for highest and lowest predicted price')
    top, bottom = get_top_and_bottom_apt(args, 16)
    top_path = get_radical_apt_most_imp_photo(args, top)
    bottom_path = get_radical_apt_most_imp_photo(args, bottom)
    plot_radical_apt_most_imp_photo(args, top_path, 16, 'highest predicted price')
    plot_radical_apt_most_imp_photo(args, bottom_path, 16, 'lowest predicted price')
