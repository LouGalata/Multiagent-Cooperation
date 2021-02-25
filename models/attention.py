import keras
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec, Layer


class SelfAttention(Layer):
    """
    Implements an self attention mechanism over time series data, weighting the
    input time series by a learned, softmax scaled attention matrix

    # Arguments
        activation: Activation function to use
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: a(x)=x)
        kernel_initializer: Initializer for the kernel weights matrix
        kernel_regularizer: Regularizer function applied to the 'kernel' weights matrix
        kernel_constraints: Constraint function applied to the 'kernel' weights matrix

    # Input shape
        3D tensor with shape (batch_size, time_step, dimensions)
    # Output shape
        3D tensor with shape (batch_size, time_step, scaled_dimensions)
    """

def __init__(self,
             activation=None,
             kernel_initializer='glorot_uniform',
             kernel_regularizer=None,
             kernel_constraints=None,
             **kwards):
    if 'input_shape' not in kwards and 'input_dim' in kwards:
        kwards['input_shape'] = (kwards.pop('input_dim'),)
    super(SelfAttention, self).__init__(**kwards)
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraints)
    self.input_spec = InputSpec(ndim=3)
    self.supports_masking = True


def build(self, input_shape):
    time_steps = input_shape[1]
    dimensions = input_shape[2]
    self.attention = keras.models.Sequential(name='attention')
    # starting off, each element of the batch is (time_steps, dimensions)
    # turn this into (time_step, 1)

    # attention matrix, this is the main thing being learned
    self.attention.add(keras.layers.Dense(dimensions,
                                          input_shape=(
                                              time_steps, dimensions,),
                                          kernel_initializer=self.kernel_initializer,
                                          kernel_regularizer=self.kernel_regularizer,
                                          kernel_constraints=self.kernel_constraints))
    self.attention.add(keras.layers.Activation(self.activation))
    # now convert to an attention vector
    self.attention.add(keras.layers.Dense(1,
                                          kernel_initializer=self.kernel_initializer,
                                          kernel_regularizer=self.kernel_regularizer,
                                          kernel_constraints=self.kernel_constraints))
    self.attention.add(keras.layers.Flatten())
    self.attention.add(keras.layers.Activation('softmax'))
    # repeat this time step weighting for each dimensions
    # RepeatVector: Repeats the input n times.
    # Example model.add(Dense(32, input_dim=32)) -->  model.output_shape == (None, 32)
    # model.add(RepeatVector(3)) --> model.output_shape == (None, 3, 32)
    self.attention.add(keras.layers.RepeatVector(dimensions))
    # reshape to be (time_steps, dimensions)
    # Permute: Permutes the dimensions of the input according to a given pattern.
    # Example: model.add(Permute((2, 1), input_shape=(10, 64))) --> model.output_shape == (None, 64, 10)
    self.attention.add(keras.layers.Permute([2, 1]))

    # noy using add_weights, so update the weights manually
    self.trainable_weights = self.attention.trainable_weights
    self.non_trainable_weights = self.attention.non_trainable_weights

    # all done
    self.built = True


def call(self, inputs):
    # build the attention matrix
    attention = self.attention(inputs)
    # apply the attention matrix with element wise multiplication
    return keras.layers.Multiply()([inputs, attention])


def compute_output_shape(self, input_shape):
    # there is no change in shape, the values are just weighted
    return input_shape


def get_config(self):
    config = {
        'activation': activations.serialize(self.attention),
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'kernel_constraints': constraints.serialize(self.kernel_constraints)
    }
    return dict(config)

