from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers.optimizers import _clip_gradients_by_norm

FLAGS = tf.app.flags.FLAGS

class NNModel(object):
  def __init__(self):
    self._global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    self._total_loss = None
    self._UPDATE_OPS_COLLECTION = ops.GraphKeys.UPDATE_OPS
    self._clip_gradients = FLAGS.max_gradient_norm

  def _get_train_op(self, clip_norm='', scope=None):
    opt = tf.train.AdamOptimizer(FLAGS.lr)
    regular_collection = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)
    total_loss = self._total_loss

    # Get Training Op
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, self._global_step)
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)
    batchnorm_updates = set(tf.get_collection(self._UPDATE_OPS_COLLECTION,
                                              scope))
    if batchnorm_updates:
      with tf.control_dependencies(batchnorm_updates):
        barrier = tf.no_op(name='update_barrier')
      total_loss = control_flow_ops.with_dependencies([barrier], total_loss)
    self._global_norm = tf.constant(0., dtype=tf.float32, name="Global_name")
    if clip_norm == 'param':
      gradients = opt.compute_gradients(total_loss)
      grads = []
      for gv in gradients:
        grads.append(gv[0])
      self._global_norm = tf.global_norm(grads)
      gradients = _clip_gradients_by_norm(gradients, self._clip_gradients)
    else:
      params = tf.trainable_variables()
      gradients = tf.gradients(total_loss, params)
      self._global_norm = tf.global_norm(gradients)
      if clip_norm == 'global':
        gradients, _ = tf.clip_by_global_norm(gradients, self._clip_gradients)
      gradients = zip(gradients, params)

    apply_gradient_op = opt.apply_gradients(gradients, self._global_step)
    self._train_op = tf.group(apply_gradient_op, variables_averages_op)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(
        var.op.name, var))
    for grad, var in gradients:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))
    self._summary_op = tf.summary.merge(summaries)

