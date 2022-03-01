import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statistics as st

CSV_PATH = '/Users/noamazmon/PycharmProjects/DL4CV/'



def main():

    min_nights = 4
    min_images = 5

    nyc_m = modify_table('{}grece listings_data.csv'.format(CSV_PATH), min_nights, min_images)

    nyc_m = nyc_m[nyc_m.mean_price < 2000]

    # scale price:
    nyc_m = scaled_price(nyc_m)

    nyc_m.to_csv('{}filtered greece listings_data.csv'.format(CSV_PATH))

    return


def plot_hist(param, param_str, title, bins_width):

    labels, counts = np.unique(param, return_counts=True)
    plt.bar(labels, counts, align='center', width=bins_width)
    plt.xticks(rotation='vertical')
    plt.xlabel(param_str)
    plt.title(title)

    return


def modify_table(path, min_nights, min_images):
    listings = pd.read_csv(path, names=['id', 'score', 'days', 'number_images'])

    # correct the score column
    listings.score = re.findall("\d+\.\d+", ' '.join(listings.score.to_list()))
    # photos
    listings = listings[listings.number_images >= min_images]
    # clean min_nights
    min_n = find_min_nights(listings)
    listings.insert(4, "min_nights", min_n, True)
    listings = listings[listings.min_nights <= min_nights]
    # get mean price
    mean_price = find_price(listings)
    listings.insert(5, "mean_price", mean_price, True)
    # get avilability
    avelability = get_avelability(listings)
    listings.insert(6, "avelability", avelability, True)
    listings = listings[listings.avelability > 0]

    print('finish')

    return listings


def find_min_nights(listing):

    min_nights = []

    for row in listing.iterrows():
        min_n = [int(re.findall("\d+", day)[0]) for day in row[1].days.split("min_nights")[1:]]
        min_nights.append(max(min_n))

    return min_nights


def find_price(listing):

    mean_prices = []

    for row in listing.iterrows():
        prices = [int(re.findall("\d+", day)[0]) for day in row[1].days.split("price")[1:]]
        mean_prices.append(st.mean(prices))

    return mean_prices


def scaled_price(listing):
    scaled_price = (listing.mean_price - st.median(listing.mean_price))/\
                   (np.quantile(listing.mean_price, 0.75)-np.quantile(listing.mean_price, 0.25))
    listing.insert(7, "scaled_price", scaled_price, True)

    return listing


def get_avelability(listing):

    availability = []

    for row in listing.iterrows():
        date_avalability = [re.findall("\w+", day)[0] == 'True' for day in row[1].days.split("avalilabe")[1:]]
        availability.append(sum(date_avalability)/len(date_avalability))

    return availability


if __name__ == "__main__":
    main()