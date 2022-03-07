import pandas as pd


def main(city_data_path, csv_to_add_photos_to):
    city_data = pd.read_csv(city_data_path, usecols=['id', 'city'])
    all_photos_data = pd.read_csv(csv_to_add_photos_to, index_col=0)
    all_photos_data = all_photos_data.join(city_data.set_index('id'), on='id')
    new_name = csv_to_add_photos_to[:csv_to_add_photos_to.find('.csv')] + '_with_cities.csv'
    all_photos_data.to_csv(new_name, index=False)

if __name__ == '__main__':
    BASE_PATH = '/Users/leeatgen/airbnb_dlp/'
    city_data_path = '{}airbnb_reg/id_and_city.csv'.format(BASE_PATH)
    csv_to_add_photos_to = '{}airbnb_reg/results/losses/predictions_07-03-22_13-46.csv'.format(BASE_PATH)
    main(city_data_path, csv_to_add_photos_to)
    print('done')