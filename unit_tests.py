import unittest
import pandas as pd
from prepare import DataFrameProcessor
from mapping import DataFrameMapper
from normalize import AddressNormalization

class TestDataFrameProcessor(unittest.TestCase):

    def setUp(self):
        data = {'A': [1, 2, None], 'B': [None, None, 3], 'C': [1, 2, 3]}
        self.df = pd.DataFrame(data)
        self.processor = DataFrameProcessor('dummy_path')
        self.processor.df = self.df

    def test_drop_columns_with_high_nas(self):
        self.processor.drop_columns_with_high_nas(threshold=0.5)
        self.assertIn('C', self.processor.df.columns)
        self.assertNotIn('B', self.processor.df.columns)

class TestDataFrameMapper(unittest.TestCase):

    def setUp(self):
        data = {'old_col': [1, 2, 3]}
        self.df = pd.DataFrame(data)
        self.mapper = DataFrameMapper(self.df, 'dummy_path')
        self.mapper.mapping_dict = {'new_col': {'features': ['old_col']}}

    def test_apply_mapping(self):
        self.mapper.apply_mapping()
        self.assertIn('new_col', self.mapper.df.columns)
        self.assertNotIn('old_col', self.mapper.df.columns)

class TestAddressNormalization(unittest.TestCase):

    def setUp(self):
        data = {'adresse': ['123 Main St', '456 Elm St', '789 Maple St']}
        self.df = pd.DataFrame(data)
        self.normalizer = AddressNormalization(self.df)

    def test_normalize_addresses(self):
        self.normalizer.normalize_addresses('adresse')
        self.assertIn('geocode_latitude', self.normalizer.df.columns)
        self.assertIn('geocode_longitude', self.normalizer.df.columns)

if __name__ == '__main__':
    unittest.main()
