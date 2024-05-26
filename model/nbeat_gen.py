import tensorflow as tf
from tensorflow.keras import layers

from model.basic_block import NBeatsBlock
from utils.utils import Utils

class NBeatsModel:
    def __init__(self, config) -> None:
        self.config = config

        self.window_size = self.config['window_size']
        self.horizon = self.config['horizon']
        self.n_epochs = self.config['n_beats_params']['epochs'] # called "Iterations" in Table 18 (N-BEATS uses 5000)
        self.n_neurons = self.config['n_beats_params']['n_neurons'] # called "Width" in Table 18
        self.n_layers = self.config['n_beats_params']['n_layers']
        self.n_stacks = self.config['n_beats_params']['n_stacks']
        self.input_size = self.window_size * self.horizon  # called "Lookback" in Table 18
        self.theta_size = self.input_size + self.horizon

        # 1. Setup N-BEATS Block layer
        self.nbeats_block_layer = NBeatsBlock(
            input_size=self.input_size,
            theta_size=self.theta_size,
            horizon=self.horizon,
            n_neurons=self.n_neurons,
            n_layers=self.n_layers,
            name="InitialBlock"
        )

        # 2. Create input to stacks
        self.stack_input = layers.Input(shape=(self.input_size,), name="stack_input")

        # 3. Create initial backcast and forecast input
        self.backcast, self.forecast = self.nbeats_block_layer(self.stack_input)
        
        # 4. Initialize residuals
        self.residuals = layers.subtract([self.stack_input, self.backcast], name="subtract_00")

        # 5. Create stacks of blocks
        for i in range(self.n_stacks - 1):  # first stack is already created in (3)
            # 6. Use the NBeatsBlock to calculate the backcast and block forecast
            self.backcast, self.block_forecast = NBeatsBlock(
                input_size=self.input_size,
                theta_size=self.theta_size,
                horizon=self.horizon,
                n_neurons=self.n_neurons,
                n_layers=self.n_layers,
                name=f"NBeatsBlock_{i}"
            )(self.residuals)
            
            # 7. Update residuals and forecast
            self.residuals = layers.subtract([self.residuals, self.backcast], name=f"subtract_{i}")
            self.forecast = layers.add([self.forecast, self.block_forecast], name=f"add_{i}")

    def build_and_train(self):
        # 8. Compile model
        self.model = tf.keras.Model(inputs=self.stack_input, outputs=self.forecast, name="model_N-BEATS")
        self.model.compile(
            loss="mae",
            optimizer=tf.keras.optimizers.Adam(self.config['n_beats_params']['lr']),
            metrics=["mae", "mse"]
        )

        # 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks

        self.utils = Utils(self.config)

        self.train_dataset, self.test_dataset = self.utils.make_tf_datasets()

        print("*"*50)
        print("Training started")


        self.model.fit(self.train_dataset,
                    epochs=self.n_epochs,
                    validation_data=self.test_dataset,
                    verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=50, verbose=1)])
        
        return self.model
    
    def test_and_evaluate(self):
        print("*"*50)
        print("testing the model")
        print("*"*50)
        self.model_preds = self.model.predict(self.test_dataset).squeeze()

        self.metrics = self.utils.evaluate_model(y_true=self.utils.y_test, y_pred=self.model_preds)

        return self.model_preds, self.metrics
    

        
