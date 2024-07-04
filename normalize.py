
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

class AddressNormalization:
    def __init__(self, dataframe):
        self.df = dataframe

    def preprocess_string_columns(self):
        """ Preprocess all string columns: convert to lowercase and remove non-alphanumeric characters """
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].str.lower().str.replace('[^a-z0-9\s]', '', regex=True).fillna('')

    def normalize_addresses(self, address_column):
        """ Normalize addresses using geocoding and add geocode columns """
        self.preprocess_string_columns()

        # Initialize geocoder
        geolocator = Nominatim(user_agent="address_normalizer")

        def geocode_address(address):
            try:
                location = geolocator.geocode(address)
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
            except GeocoderTimedOut:
                return None, None

        # Apply geocoding and create new columns for latitude and longitude
        #self.df['normalized_address'] = self.df[address_column].apply(lambda x: geocode_address(x)[0])
        self.df['geocode_latitude'] = self.df[address_column].apply(lambda x: geocode_address(x)[0])
        self.df['geocode_longitude'] = self.df[address_column].apply(lambda x: geocode_address(x)[1])

    def export_results(self, output_path):
        """ Export the DataFrame with normalized addresses and geocodes to a CSV file """
        self.df.to_csv(output_path, index=False)
        print(f"Data exported to {output_path}")

