import pandas as pd
import os
from tabulate import tabulate

def process_csv_files(input_folder):
    # Function to infer and rename data types
    def infer_and_rename_dtype(series):
        dtype = pd.api.types.infer_dtype(series)
        if dtype in ['integer', 'mixed-integer']:
            return 'Integer'
        elif dtype in ['floating', 'mixed-integer-float']:
            return 'Float'
        elif dtype in ['string', 'unicode', 'bytes']:
            return 'String'
        elif dtype == 'boolean':
            return 'Boolean'
        elif dtype == 'datetime':
            return 'Datetime'
        elif dtype in ['timedelta']:
            return 'Timedelta'
        elif dtype == 'complex':
            return 'Complex'
        else:
            return 'Unknown'

    output_excel_path = os.path.join(input_folder, 'synthetic_tables.xlsx')

    # Create a writer object to write multiple sheets to the same Excel file
    with pd.ExcelWriter(output_excel_path) as writer:
        # Loop through all files in the folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(input_folder, file_name)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Create a synthetic table
                synthetic_table = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': [infer_and_rename_dtype(df[col]) for col in df.columns],
                    'Description': [''] * len(df.columns)  # Placeholder for descriptions
                })

                # Write the synthetic table to the Excel file
                sheet_name = os.path.splitext(file_name)[0]
                synthetic_table.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Synthetic tables have been written to {output_excel_path}.")



class DataFrameDescriber:
    def __init__(self, dataframe):
        self.df = dataframe

    def print_formatted(self, data, title):
        """Prints data in a table format with a title."""
        print(title)
        print(tabulate(data, headers='keys', tablefmt='psql'))
        print("\n")

    def get_info(self):
        """Print formatted information about DataFrame including the index dtype and columns, non-null values and memory usage."""
        buf = pd.io.StringIO()
        self.df.info(buf=buf)
        info = buf.getvalue()
        print("DataFrame Information:\n", info)

    def get_description(self, include='all'):
        """Generate and print formatted descriptive statistics."""
        description = self.df.describe(include=include)
        self.print_formatted(description, "Descriptive Statistics:")

    def get_missing_values(self):
        """Print formatted count of missing values per column."""
        missing_values = self.df.isnull().sum()
        self.print_formatted(missing_values, "Missing Values Count per Column:")

    def get_missing_values_percentage_by_column(self):
        """Print formatted percentage of missing values per column."""
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        self.print_formatted(missing_percentage, "Missing Values Percentage per Column:")

    def get_missing_values_percentage_by_row(self):
        """Print formatted percentage of missing values per row."""
        missing_percentage_rows = (self.df.isnull().sum(axis=1) / self.df.shape[1]) * 100
        self.print_formatted(missing_percentage_rows.describe(), "Missing Values Percentage per Row Statistics:")

    def get_unique_values(self):
        """Print formatted number of unique values per column."""
        unique_values = self.df.nunique()
        self.print_formatted(unique_values, "Unique Values per Column:")

    def get_correlation(self):
        """Print formatted correlation matrix of the DataFrame."""
        correlation = self.df.corr()
        self.print_formatted(correlation, "Correlation Matrix:")

    def export_to_excel(self, output_path):
        """Export all descriptions and statistics to an Excel file."""
        with pd.ExcelWriter(output_path) as writer:
            self.df.describe(include='all').to_excel(writer, sheet_name='Descriptive Statistics')
            self.df.isnull().sum().to_frame('Missing Values Count').to_excel(writer, sheet_name='Missing Values Count')
            ((self.df.isnull().sum() / len(self.df)) * 100).to_frame('Missing Values Percentage').to_excel(writer, sheet_name='Missing Values Percentage')
            self.df.nunique().to_frame('Unique Values').to_excel(writer, sheet_name='Unique Values')
            self.df.corr().to_excel(writer, sheet_name='Correlation Matrix')
            print(f"Report exported to {output_path}")