import argparse
import matplotlib.pyplot as plt
import pandas as pd
import get_apt_city
import numpy as np

# ----------------------------------------------------------------------
# Parameters
parser = argparse.ArgumentParser(description='airbnb_reg: Photo album Event recognition using Transformers Attention.')
parser.add_argument('--results_path', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/results')
parser.add_argument('--path_output', type=str, default='/home/labs/testing/class63/airbnb_dlp/airbnb_reg/outputs/')
parser.add_argument('--job_id', type=str, default='08-03-22_11-33-20')
parser.add_argument('--base_path', type=str, default='/Users/noamazmon/airbnb_dlp/')
parser.add_argument('--pred_path', type=str, default='airbnb_reg/results/losses/predictions_07-03-22_15-11.csv')
parser.add_argument('--att_path', type=str, default='airbnb_reg/outputs/att_data_08-03-22_11-33-20.csv')
parser.add_argument('--id_city_path', type=str, default='airbnb_reg/id_and_city.csv')

def count_I0(att_data_path, args):

    att_data = pd.read_csv(att_data_path)

    sig_im = att_data['most_important_pic_path']
    sig_im_lists = sig_im.str.split("/")
    most_important_pic = []
    for l in sig_im_lists:
        most_important_pic.append(l[-1].split(".")[0][-2:])
    att_data['most_important_pic'] = most_important_pic
    is_I0 = (att_data['most_important_pic'] == "I0")
    att_data['is_I0'] = is_I0

    guessed_baseline = att_data.is_I0.sum()
    guessed_baseline = guessed_baseline / len(is_I0)

    return att_data, guessed_baseline

def get_pred_dist(args):

    predictions_path = args.results_path + f"/losses/predictions_{args.job_id}.csv"
    predictions = pd.read_csv(predictions_path)
    pred = predictions['pred']
    gt = predictions['price']
    pred.plot(kind='hist')
    plt.show()
    gt.loc[gt<=2].plot(kind='hist')
    plt.show()


def get_loss_per_city(args):
    pred_full_path = args.base_path + args.pred_path
    city_id_full_path = args.base_path + args.id_city_path
    pred_with_city_df = get_apt_city.add_city_per_id(city_id_full_path, pred_full_path)
    pred_with_city_df['loss'] = (pred_with_city_df['price'] - pred_with_city_df['pred'])**2
    avg_loss_per_city = pred_with_city_df.groupby(['city'], as_index=False).median()
    return avg_loss_per_city


def from_loss_to_usd(loss, perc_75, perc_25):
    return ((loss ** 0.5) * ( perc_75 - perc_25)) + perc_25


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


def plot_loss_per_city_and_base_line(args,avg_loss_per_city):
    width = 0.35
    labels_short = avg_loss_per_city['city']
    labels = []
    citys = {'ber': 'Berlin', 'nyc': 'New York', 'ist': 'Istanbul','tor':'Toronto', 'gre':'Athens'}
    for city in labels_short:
        labels.append(citys[city])
    x = np.arange(avg_loss_per_city.shape[0])
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, avg_loss_per_city['loss_in_usd'], width, label='loss')
    rects2 = ax.bar(x + width / 2, avg_loss_per_city['median_baseline_in_usd'], width, label='baseline')
    ax.set_ylabel('USD $')
    ax.set_title('Loss in USD per city')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            height = round(height,1)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  fontsize= 8,
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig('{}{}_loss_per_city.jpg'.format(args.base_path ,args.pred_path), dpi=300)
    plt.show()


def main(args):
    att_data_path = args.path_output + f"/att_data_{args.job_id}.csv"
    att_data, guessed_baseline = count_I0(att_data_path, args=args)
    print(f"predicted the first image {guessed_baseline} time")

    get_pred_dist(args)

    print('Done\n')


if __name__ == '__main__':
    args = parser.parse_args()
    city_conv_data = {'nyc': {'0.25': 126.431266846361, '0.75': 248.517520215633, 'mean_baseline_loss': 1.5701, 'median_baseline_loss':1.4448 },
                      'ist': {'0.25': 33.9339622641509, '0.75': 100.161725067385,  'mean_baseline_loss': 0.5891, 'median_baseline_loss': 0.5499 },
                      'ber': {'0.25': 79.9245283018868, '0.75': 156.0,  'mean_baseline_loss':0.9418, 'median_baseline_loss':0.8731 },
                      'tor': {'0.25': 55.3793800539084, '0.75': 143.995956873315,  'mean_baseline_loss': 0.9461, 'median_baseline_loss':0.8662},
                      'gre': {'0.25': 44.2681940700809, '0.75': 89.0,  'mean_baseline_loss': 1.4517, 'median_baseline_loss':1.3287}}


    avg_loss_per_city = get_loss_per_city(args)
    avg_loss_per_city = add_baselines_and_convert_to_usd(city_conv_data, avg_loss_per_city)
    plot_loss_per_city_and_base_line(args, avg_loss_per_city)

    main(args)
