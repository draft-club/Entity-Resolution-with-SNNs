import pandas as pd
import json


class DataFrameProcessor:
    def __init__(self, input_path):
        self.input_path = input_path
        self.df = pd.read_csv(input_path)

    def drop_columns_with_high_nas(self, threshold=0.6):
        """ Drop columns from DataFrame with more than `threshold` NaN values """
        na_counts = self.df.isna().sum()
        columns_to_drop = na_counts[na_counts / len(self.df) > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with more than {threshold * 100}% NaN values.")

    def filter_and_save_df(self, columns_dict, output_path):
        """
        Filters the DataFrame to only include columns that are keys in the provided dictionary.
        The filtered DataFrame is then saved to the specified output path.

        Args:
        columns_dict (dict): A dictionary where the keys are the column names to keep.
        output_path (str): The path where the filtered DataFrame will be saved.

        Returns:
        None
        """
        # Filter the DataFrame to keep only the columns that are keys in the dictionary
        filtered_cols = [col for col in self.df.columns if col in columns_dict.keys()]

        if not filtered_cols:
            # If no columns match, output a warning and break the execution
            raise ValueError("No columns in the DataFrame correspond to the keys in the dictionary.")

        # Create a new DataFrame with the filtered columns
        filtered_df = self.df[filtered_cols]

        # Save the filtered DataFrame to the specified output path
        if output_path.endswith('.csv'):
            filtered_df.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            filtered_df.to_excel(output_path, index=False)
        else:
            raise ValueError("Output file format not supported. Please use .csv or .xlsx.")

        print(f"Filtered DataFrame saved to {output_path}")

