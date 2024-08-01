import pandas as pd

class DataFrameProcessor:
    """
    Class to process DataFrame by dropping columns with high NaNs.
    """

    def __init__(self, input_path):
        self.input_path = input_path
        self.df = pd.read_csv(input_path)

    def drop_columns_with_high_nas(self, threshold=0.6):
        """
        Drop columns from DataFrame with more than `threshold` NaN values.

        Args:
            threshold (float): Proportion of NaN values to drop the column.
        """
        na_counts = self.df.isna().sum()
        columns_to_drop = na_counts[na_counts / len(self.df) > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with more than {threshold * 100}% NaN values.")
