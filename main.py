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


def prepare_datasets(pairs_train, labels_train, pairs_val, labels_val, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(((pairs_train[:, 0], pairs_train[:, 1]), labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(((pairs_val[:, 0], pairs_val[:, 1]), labels_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def main():
    # Check GPU availability
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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
    describer.get_info()
    describer.get_description()
    describer.get_missing_values()
    describer.get_missing_values_percentage_by_column()
    describer.get_missing_values_percentage_by_row()
    describer.get_unique_values()


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
