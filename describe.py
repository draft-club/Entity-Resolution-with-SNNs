import pandas as pd
from tabulate import tabulate

class DataFrameDescriber:
    """
    Class to describe DataFrame with statistics, missing values, and unique values.
    """

    def __init__(self, dataframe):
        self.df = dataframe

    def _print_formatted(self, data, title):
        """Prints data in a table format with a title."""
        print(title)
        print(tabulate(data, headers='keys', tablefmt='psql'))
        print("\n")

    def describe_all(self):
        """Run all descriptive methods and print results."""
        self.get_info()
        self.get_description()
        self.get_missing_values()
        self.get_missing_values_percentage_by_column()
        self.get_missing_values_percentage_by_row()
        self.get_unique_values()

    def get_info(self):
        """Print formatted information about DataFrame including the index dtype and columns, non-null values and memory usage."""
        buf = pd.io.StringIO()
        self.df.info(buf=buf)
        info = buf.getvalue()
        print("DataFrame Information:\n", info)

    def get_description(self, include='all'):
        """Generate and print formatted descriptive statistics."""
        description = self.df.describe(include=include)
        self._print_formatted(description, "Descriptive Statistics:")

    def get_missing_values(self):
        """Print formatted count of missing values per column."""
        missing_values = self.df.isnull().sum()
        self._print_formatted(missing_values, "Missing Values Count per Column:")

    def get_missing_values_percentage_by_column(self):
        """Print formatted percentage of missing values per column."""
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        self._print_formatted(missing_percentage, "Missing Values Percentage per Column:")

    def get_missing_values_percentage_by_row(self):
        """Print formatted percentage of missing values per row."""
        missing_percentage_rows = (self.df.isnull().sum(axis=1) / self.df.shape[1]) * 100
        self._print_formatted(missing_percentage_rows.describe(), "Missing Values Percentage per Row Statistics:")

    def get_unique_values(self):
        """Print formatted number of unique values per column."""
        unique_values = self.df.nunique()
        self._print_formatted(unique_values, "Unique Values per Column:")

    def export_to_excel(self, output_path):
        """Export all descriptions and statistics to an Excel file."""
        with pd.ExcelWriter(output_path) as writer:
            self.df.describe(include='all').to_excel(writer, sheet_name='Descriptive Statistics')
            self.df.isnull().sum().to_frame('Missing Values Count').to_excel(writer, sheet_name='Missing Values Count')
            ((self.df.isnull().sum() / len(self.df)) * 100).to_frame('Missing Values Percentage').to_excel(writer, sheet_name='Missing Values Percentage')
            self.df.nunique().to_frame('Unique Values').to_excel(writer, sheet_name='Unique Values')
            self.df.corr().to_excel(writer, sheet_name='Correlation Matrix')
            print(f"Report exported to {output_path}")
