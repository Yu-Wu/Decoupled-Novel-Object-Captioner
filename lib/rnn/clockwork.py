from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np
from ltf.lib.rnn import rnn_cell

FLAGS = tf.app.flags.FLAGS
RNNCell = rnn_cell.RNNCell

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class ClockWorkLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, input_size=None, activation=tanh,
               scope=None, forget_bias=1.0, state_is_tuple=False):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    assert state_is_tuple == False
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    # periods = [1, 3, 6]
    periods = [1, 2, 4]
    n = int(math.ceil(1. * self._num_units / len(periods)))
    self._mask4 = np.zeros((self._num_units, self._num_units * 4), np.float32)
    self._period = np.zeros((self._num_units, ), np.int32)
    for i, T in enumerate(periods):
      for offset in xrange(4):
        tmp_s = self._num_units * offset + i * n
        if False:
          self._mask4[: (i + 1) * n, tmp_s: tmp_s + n] = 1.
        else:
          self._mask4[i * n:, tmp_s: tmp_s + n] = 1.

      self._period[i * n: (i + 1) * n] = T
    self._scope = scope or type(self).__name__
    with vs.variable_scope(self._scope+"_Var"):
      self._hidden_state_g_w = tf.get_variable("state_g_w", [self._num_units, self._num_units * 4])
      self._Bgh = tf.get_variable("g_b", [self._num_units * 4],
                                                initializer=tf.constant_initializer(
                                                0., dtype=tf.float32))
    self._Wgh = tf.mul(self._hidden_state_g_w, self._mask4)

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    step_t, state = state
    with vs.variable_scope(self._scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(1, 2, state)
      ix, jx, fx, ox = array_ops.split(1, 4, rnn_cell._linear([inputs], 4 * self._num_units, False))
      ih, jh, fh, oh = array_ops.split(1, 4, tf.matmul(h, self._Wgh) + self._Bgh)
      i, j, f, o = ix + ih, jx + jh, fx + fh, ox + oh

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      # i, j, f, o = array_ops.split(1, 4, concat)

      active = (step_t % self._period) == 0
      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_c = active * new_c + (1 - active) * c
      new_h = self._activation(new_c) * sigmoid(o)
      new_h = active * new_h + (1 - active) * h

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat(1, [new_c, new_h])
      return new_h, [new_state]


class ClockWorkGRUCell(RNNCell):
  def __init__(self, num_units, input_size=None, activation=tanh, scope=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

    # periods = [1, 3, 6, 10]
    periods = [1, 3, 6]
    # periods = [1, 2, 4, 8]
    n = int(math.ceil(1. * self._num_units / len(periods)))
    self._mask = np.zeros((self._num_units, self._num_units), np.float32)
    self._mask2 = np.zeros((self._num_units, self._num_units * 2), np.float32)
    self._period = np.zeros((self._num_units, ), np.int32)
    for i, T in enumerate(periods):
      tmp_s = self._num_units + i * n
      if False:
        self._mask[i * n:, i * n: (i + 1) * n] = 1.
        self._mask2[i * n:, i * n: (i + 1) * n] = 1.
        self._mask2[i * n:, tmp_s: tmp_s + n] = 1.
      else:
        self._mask[: (i + 1) * n, i * n: (i + 1) * n] = 1.
        self._mask2[: (i + 1) * n, i * n: (i + 1) * n] = 1.
        self._mask2[: (i + 1) * n, tmp_s: tmp_s + n] = 1.

      self._period[i * n: (i + 1) * n] = T

    self._scope = scope or type(self).__name__
    with vs.variable_scope(self._scope+"_Var"):  # "GRUCell"
      self._mask = tf.constant(self._mask, dtype=tf.float32, name="state_mask")
      self._mask2 = tf.constant(self._mask2, dtype=tf.float32, name="state_mask2")
      # self._period = tf.constant(self._period, dtype=tf.int32, name="period")

      self._hidden_state_g_w = tf.get_variable("state_g_w", [self._num_units, self._num_units * 2])
      self._Bgh = tf.get_variable("g_b", [self._num_units * 2],
                                               initializer=tf.constant_initializer(
                                                   1., dtype=tf.float32))
      self._hidden_state_c_w = tf.get_variable("state_c_w", [self._num_units, self._num_units])
      self._Bch = tf.get_variable("c_b", [self._num_units],
                                               initializer=tf.constant_initializer(
                                                   0., dtype=tf.float32))
      if FLAGS.phase_train and False:
        dropout_ratio = 0.5
        self._mask2 = tf.nn.dropout(self._mask2, dropout_ratio)
        self._mask = tf.nn.dropout(self._mask, dropout_ratio)
      self._Wgh = tf.multiply(self._hidden_state_g_w, self._mask2)
      self._Wch = tf.multiply(self._hidden_state_c_w, self._mask)

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    step_t, state = state
    with vs.variable_scope(self._scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates_X"):
        rx, ux = tf.split(rnn_cell._linear([inputs],
                                                2 * self._num_units, False), 2, 1)
        rh, uh = tf.split(tf.matmul(state, self._Wgh) + self._Bgh, 2, 1)
        r, u = rx + rh, ux + uh
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        cx = rnn_cell._linear([inputs], self._num_units, False)
        c = cx + tf.matmul(state * r, self._Wch) + self._Bch
        c = self._activation(c)
      new_h = u * state + (1 - u) * c
    active = (step_t % self._period) == 0
    new_h = active * new_h + (1 - active) * state
    return new_h, [new_h]

  def __call__2(self, inputs, state, scope=None):
    step_t, state = state
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, rnn_cell._linear([inputs, state],
                                             2 * self._num_units, True, 1.0))
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        c = self._activation(rnn_cell._linear([inputs, r * state],
                                     self._num_units, True))
      new_h = u * state + (1 - u) * c
    active = (step_t % self._period) == 0
    new_h = active * new_h + (1 - active) * state
    return new_h, [new_h]

  def __call__0(self, inputs, state, scope=None):
    step_t, state = state
    with vs.variable_scope(self._scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates_X"):
        x = rnn_cell._linear([inputs],
                             self._num_units, True, 0.)
        h = tf.matmul(state, self._Wch) + self._Bch
        new_h = self._activation(x + h)
    active = (step_t % self._period) == 0
    new_h = active * new_h + (1 - active) * state
    return new_h, [new_h]
