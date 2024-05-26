import json
from tensorflow.keras.utils import plot_model

from model.nbeat_gen import NBeatsModel

with open('config/config.json') as f:
    config = json.load(f)

nbeats_model = NBeatsModel(config)

# Train the model
model = nbeats_model.build_and_train()


# Test the trained model on the test dataset
model_preds, metrics = nbeats_model.test_and_evaluate()
print(f"Test results = {metrics}")


# Save the model
model.save(f"{config['model_dir']}/nbeats.keras")

# Save the model architecture
plot_model(model, to_file=f"{config['images_dir']}/nbeats_generic.png", show_shapes=True, show_layer_names=True)
