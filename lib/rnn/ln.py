import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import array_ops
from ltf.lib.rnn import rnn_cell
from tensorflow.contrib import losses


def ln(_input, s, b, epsilon=1e-1, max=1000):
  """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
  m, v = tf.nn.moments(_input, [1], keep_dims=True)
  normalised_input = (_input - m) / tf.sqrt(v + epsilon)
  return normalised_input * s + b


class LNGRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      print("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    dim = self._num_units
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.

        s1 = tf.get_variable("s1", [2 * dim], initializer=tf.constant_initializer(1.0))
        s2 = tf.get_variable("s2", [2 * dim], initializer=tf.constant_initializer(1.0))
        s3 = tf.get_variable("s3", [dim], initializer=tf.constant_initializer(1.0))
        s4 = tf.get_variable("s4", [dim], initializer=tf.constant_initializer(1.0))
        b1 = tf.get_variable("b1", [2 * dim], initializer=tf.constant_initializer(0.))
        b2 = tf.get_variable("b2", [2 * dim], initializer=tf.constant_initializer(0.))
        b3 = tf.get_variable("b3", [dim], initializer=tf.constant_initializer(0.))
        b4 = tf.get_variable("b4", [dim], initializer=tf.constant_initializer(0.))

        input_below_ = rnn_cell._linear([inputs],
                               2 * self._num_units, False, scope="out_1")
        input_below_ = ln(input_below_, s1, b1)
        state_below_ = rnn_cell._linear([state],
                               2 * self._num_units, False, scope="out_2")
        state_below_ = ln(state_below_, s2, b2)
        out = tf.add(input_below_, state_below_)
        r, u = array_ops.split(1, 2, out)
        r, u = sigmoid(r), sigmoid(u)

      with vs.variable_scope("Candidate"):
        input_below_x = rnn_cell._linear([inputs],
                                        self._num_units, False, scope="out_3")
        input_below_x = ln(input_below_x, s3, b3)
        state_below_x = rnn_cell._linear([state],
                                        self._num_units, False, scope="out_4")
        state_below_x = ln(state_below_x, s4, b4)
        c_pre = tf.add(input_below_x, r * state_below_x)
        c = self._activation(c_pre)
      new_h = u * state + (1 - u) * c
    return new_h, new_h
