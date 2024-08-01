import os
import pandas as pd
import tensorflow as tf
from prepare import DataFrameProcessor
from mapping import DataFrameMapper
from describe import DataFrameDescriber
from normalize import AddressNormalization
from snn_model import SiameseNetwork
from tuner import SNNTuner
from constants import INPUT_FOLDER, OUTPUT_FOLDER, DICTIONARY_FOLDER

def gpu_check(func):
    """
    Decorator to check the availability of GPUs before executing a function.
    """
    def wrapper(*args, **kwargs):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        return func(*args, **kwargs)
    return wrapper

def prepare_datasets(pairs_train, labels_train, pairs_val, labels_val, batch_size):
    """
    Prepares training and validation datasets for mini-batch training.

    Args:
        pairs_train (numpy.ndarray): Training data pairs.
        labels_train (numpy.ndarray): Training data labels.
        pairs_val (numpy.ndarray): Validation data pairs.
        labels_val (numpy.ndarray): Validation data labels.
        batch_size (int): Size of the mini-batch.

    Returns:
        tuple: TensorFlow datasets for training and validation.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(((pairs_train[:, 0], pairs_train[:, 1]), labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(((pairs_val[:, 0], pairs_val[:, 1]), labels_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset

@gpu_check
def main():
    """
    Main function to execute the data processing and model training pipeline.
    """
    input_path = os.path.join(INPUT_FOLDER, 'filtered_mapped_transformed_contribuable_last_call.csv')
    dict_path = os.path.join(DICTIONARY_FOLDER, 'mapping_dts.json')
    output_path = os.path.join(OUTPUT_FOLDER, 'output.csv')

    # Prepare Data
    processor = DataFrameProcessor(input_path)
    processor.drop_columns_with_high_nas()

    # Mapping
    mapper = DataFrameMapper(processor.df, dict_path)
    mapper.apply_mapping()
    mapper.filter_columns()

    # Describe
    describer = DataFrameDescriber(mapper.df)
    describer.describe_all()

    # Normalize
    normalizer = AddressNormalization(mapper.df)
    normalizer.normalize_addresses('adresse')
    normalizer.export_results(output_path)

    # Load Data for SNN
    df = pd.read_csv(output_path)
    input_shape = (df.shape[1], 1)

    # Assuming pairs_train, labels_train, pairs_val, labels_val are prepared
    # Replace with actual data preparation code
    pairs_train, labels_train, pairs_val, labels_val = None, None, None, None

    # Prepare datasets for mini-batch training
    batch_size = 128
    train_dataset, val_dataset = prepare_datasets(pairs_train, labels_train, pairs_val, labels_val, batch_size)

    # Hyperparameter Tuning
    tuner = SNNTuner(input_shape)
    best_hps = tuner.run_tuning(pairs_train, labels_train, pairs_val, labels_val)

    # Create and Train SNN Model with Best Hyperparameters
    snn = SiameseNetwork(input_shape)
    model = snn.build_model(best_hps)
    snn.train(model, train_dataset, val_dataset)

if __name__ == '__main__':
    main()
