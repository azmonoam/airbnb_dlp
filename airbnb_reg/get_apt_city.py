import pandas as pd


def add_city_per_id(city_data_path, csv_to_add_city_to):
    city_data = pd.read_csv(city_data_path, usecols=['id', 'city'])
    all_photos_data = pd.read_csv(csv_to_add_city_to, index_col=0)
    all_photos_data = all_photos_data.join(city_data.set_index('id'), on='id')
    new_name = csv_to_add_city_to[:csv_to_add_city_to.find('.csv')] + '_with_cities.csv'
    all_photos_data.to_csv(new_name, index=False)
    return all_photos_data.reset_index()

if __name__ == '__main__':
    BASE_PATH = '/Users/leeatgen/airbnb_dlp/'
    city_data_path = '{}airbnb_reg/id_and_city.csv'.format(BASE_PATH)
    csv_to_add_city_to = '{}airbnb_reg/results/losses/predictions_07-03-22_13-46.csv'.format(BASE_PATH)
    df_with_city = add_city_per_id(city_data_path, csv_to_add_city_to)
    print('done')