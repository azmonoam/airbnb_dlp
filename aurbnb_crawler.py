'''
To use this file, use:

with open('listings_path', mode='r') as f:          Example: 'listings_path' = 'istanbul/airbnb Istanbul - ist_2020.csv'
    reader = csv.reader(f)
    listing_ids = [row[0] for row in reader]

out_file = 'listings_extracted_data_path'         Example: 'listings_extracted_data_path' = 'istanbul/listings_data.csv'

out_folder = 'images_out_path'                          Example: 'images_out_path' = './istanbul/airbnb_images_istanbul'
crawler(listing_ids, out_folder, out_file)
'''
from bs4 import BeautifulSoup
import os
import requests
from tqdm import tqdm
import pandas as pd
import airbnb
import time
from csv import writer

def crawler(listing_ids, out_folder, file_name):
    listings = pd.DataFrame(columns=['id', 'score', 'dates', 'num_images'])

    for idx, id_number in enumerate(tqdm(listing_ids)):
        out = airbnb_scarpe(id_number, out_folder, idx)
        if out:
            score = out[0]
            num_images = out[1]

            seen_dates = []
            apartment_dates = {}

            for i in range(2):
                try:
                    api = airbnb.Api(randomize=True)
                    cal = api.get_calendar(str(id_number))
                    break
                except:
                    time.sleep(30)

            months = [month['days'] for month in cal['calendar_months']]

            for month in months:
                for entry in month:
                    date = entry['date']
                    if date not in seen_dates:
                        seen_dates.append(date)
                        avalilabe = entry['available']
                        price = entry['price']['local_price_formatted']
                        min_nights = entry['min_nights']
                        max_nights = entry['max_nights']
                        apartment_dates[date] = {'avalilabe': avalilabe, 'price': price, 'min_nights': min_nights, 'max_nights':max_nights}

            # listings = listings.append({'id': id_number, 'score': score, 'dates': apartment_dates, 'num_images':num_images}, ignore_index=True)
            list_of_elem = [id_number, score, apartment_dates, num_images]
            with open(file_name, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(list_of_elem)

    print(listings)


def get_images(soup, id_number, prefix, out_folder):
    images = 0
    for img_idx, image in enumerate(soup.findAll("img")):
        filename = image["src"].split("/")[-1]
        r = requests.get(prefix + filename)
        type = r.headers['Content-Type'].split('/')
        type_class = type[0]
        if type_class == 'image':
            format = type[-1]
            outpath = os.path.join(out_folder, 'A' + str(id_number) + '_I' + str(img_idx) + '.'+format)
            with open(outpath, 'wb') as f:
                images += 1
                f.write(r.content)

    return images


def get_score(soup):

    try:
        score = soup.body.find_all(class_='_12si43g')[0].contents[0]
    except:
        score = False
    return score


def airbnb_scarpe(id_number, out_folder, apartment_idx):
    """Downloads all the images at 'url' to /test/"""
    url = 'https://www.airbnb.com/rooms/{id_number}/'.format(id_number=id_number)
    answer = requests.get(url)
    page_content = answer.content
    if page_content:
        soup = BeautifulSoup(page_content, features='html.parser')

        # Get Score
        score = get_score(soup)

        # Download Images
        prefix = 'https://a0.muscache.com/im/pictures/'
        images = get_images(soup, id_number, prefix, out_folder)

    return [score, images] if score and images else False
