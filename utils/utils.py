import tensorflow as tf
import pandas as pd
import numpy as np
import subprocess
import os

class Utils():
    def __init__(self, config) -> None:
        self.config = config
        self.data_dir = self.config['data_dir']
        self.image_dir = self.config['images_dir']
        self.model_dir = self.config['model_dir']

        # Create directories for data, models, and images
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _download_csv_data(self):

        file_id = self.config['file_id']
        output_name = self.config['output_name']

        # Construct the URL and the complete output file name
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        self.output_filename = f"{self.data_dir}/{output_name}.csv"

        # Construct the wget command as a list of arguments
        command = ["wget", "--no-check-certificate", url, "-O", self.output_filename]

        # Execute the command
        subprocess.run(command)

    def _create_features_and_labels(self):
        '''
        This functions create features and labels from a csv file.
        It add the windowed columns and split data into train and test sets
        Parameters: csv file name, window size, test size
        Return: features and labels for train and test sets
        '''

        # Download the data
        self._download_csv_data()

        # Read the data
        df = pd.read_csv(self.output_filename, parse_dates=['timestamp'], index_col=['timestamp'])

        # As N_BEATS is used for univariate time series, I will use temp column for demonstration purpose
        temp_nbeats  =df[['temp']].copy()

        # Add windowed columns
        for i in range(self.config['window_size']):
            temp_nbeats[f"temp+{i+1}"] = temp_nbeats["temp"].shift(periods=i+1)
        
        # Make features and labels
        X = temp_nbeats.dropna().drop("temp", axis=1)
        y = temp_nbeats.dropna()["temp"]

        # Make train and test sets
        split_size = int(len(X) * (1 - self.config['test_size']))
        X_train, y_train = X[:split_size], y[:split_size]
        X_test, y_test = X[split_size:], y[split_size:]
        
        return X_train, X_test, y_train, y_test
        

    def make_tf_datasets(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self._create_features_and_labels()

        # 1. Turn train and test arrays into tensor Datasets
        train_features_dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
        train_labels_dataset = tf.data.Dataset.from_tensor_slices(self.y_train)

        test_features_dataset = tf.data.Dataset.from_tensor_slices(self.X_test)
        test_labels_dataset = tf.data.Dataset.from_tensor_slices(self.y_test)

        # 2. Combine features & labels
        train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
        test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

        # 3. Batch and prefetch for optimal performance
        # The batch size is 1024. Ref.from Appendix D in N-BEATS paper
        batch_size = self.config['n_beats_params']['batch_size']
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset
    
    def evaluate_model(self, y_true, y_pred):
        # Make sure float32 (for metric calculations)
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Calculate various metrics
        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        mae_metric.update_state(y_true, y_pred)
        mae = mae_metric.result()

        mse_metric = tf.keras.metrics.MeanSquaredError()
        mse_metric.update_state(y_true, y_pred)
        mse = mse_metric.result()
        rmse = tf.sqrt(mse)

        # Account for different sized metrics (for longer horizons, reduce to single number)
        if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
            mae = tf.reduce_mean(mae)
            rmse = tf.reduce_mean(rmse)

        return {"mae": mae.numpy(), "rmse": rmse.numpy()}
    

