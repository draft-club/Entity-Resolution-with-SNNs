
# Siamese Neural Network (SNN) for Entity Resolution

This project implements a Siamese Neural Network (SNN) for entity resolution, specifically designed to handle complex data matching scenarios. The project is structured to prepare, map, normalize, and process datasets, followed by training a Siamese Neural Network model using TensorFlow.


## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- Python 3.7+
- pip (Python package installer)
- Git (for cloning the repository)
- Docker (optional, for containerization)

## Installation

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository

Clone the project repository from GitHub:

```bash
git clone https://github.com/yourusername/snn-entity-resolution.git
cd snn-entity-resolution
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create a virtual environment to manage your dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Python Packages
Install the dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Configure Paths and Variables
Review and configure the config.yaml and constants.py files to ensure the paths and variables match your environment:

config.yaml: Set paths for input and output data directories.
constants.py: Adjust constants used across the project, such as folder paths and thresholds.
Usage
1. Prepare Data
Ensure your input data (CSV files) is placed in the input_data/ directory. The data processing pipeline will read these files, clean them, and prepare them for further analysis.

2. Run the Data Pipeline
Execute the main script to run the entire data processing and model training pipeline:

```bash
python main.py
```

This script performs the following steps:

- Data Preparation: Cleans and filters the dataset, dropping columns with a high percentage of missing values.
- Data Mapping: Maps the dataset columns according to a predefined dictionary.
- Data Normalization: Normalizes address columns using geocoding to ensure consistency.
- Data Description: Generates descriptive statistics, identifies missing values, and provides a summary of unique values.
- Model Training: Prepares datasets for mini-batch training and trains the Siamese Neural Network model using TensorFlow.
### 3. Monitor Model Training
The script will output progress and results to the console. If GPUs are available, the script will utilize them to accelerate training.

### 4. Review Output
Processed data files and model outputs will be saved in the output_data/ directory.

Running Tests
Unit tests are included to ensure that the various components of the project work as expected.

To run all tests, execute the following command:
```bash
python -m unittest discover
```

This will automatically discover and run all unit tests located in the project directory.

# Containerization with Docker (Optional)
You can containerize the project using Docker. This ensures that the application runs in a consistent environment, regardless of where it is deployed.

## 1. Build the Docker Image
Build the Docker image using the provided Dockerfile:
```bash
docker build -t snn-entity-resolution .
```

## 2. Run the Docker Container
Run the container with the following command:

```bash
docker run --rm -v $(pwd)/input_data:/app/input_data -v $(pwd)/output_data:/app/output_data snn-entity-resolution
```

This command maps your local input_data and output_data directories to the corresponding directories in the container.
