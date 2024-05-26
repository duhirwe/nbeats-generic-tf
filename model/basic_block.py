import tensorflow as tf

# Create NBEATBlock custom layer
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self,
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)

    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [tf.keras.layers.Dense(self.n_neurons, activation='relu') for _ in range(self.n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = tf.keras.layers.Dense(self.theta_size, activation='linear', name='theta')



  def call(self, inputs): #  the call method is what runs when the layer is called
    self.x = inputs
    for layer in self.hidden: # pass inputs through each hidden layer
      self.x = layer(self.x)
    self.theta = self.theta_layer(self.x)

    # Output the backcast and forecast from theta
    self.backcast, self.forecast = self.theta[:, :self.input_size], self.theta[:, -self.horizon:]
    return self.backcast, self.forecast