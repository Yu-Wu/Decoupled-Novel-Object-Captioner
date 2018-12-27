import json
import numpy as np
import os, math, time
import tensorflow as tf

import multiprocessing as mp
from multiprocessing import Queue

FLAGS = tf.app.flags.FLAGS

'''
Part code is modified from tflearn.
'''
class DataLoader(object):
  def _set_num_instances(self):
    self._num_instances = len(self._dataset[FLAGS.stage])

  def __init__(self, coord):
    self._coord = coord
    with open(FLAGS.dataset_path, 'r') as fin:
      self._dataset = json.load(fin)

    self._phase_train = FLAGS.stage == "train"
    self._set_num_instances()
    self._batch_size = FLAGS.batch_size
    self._num_threads = FLAGS.num_loaders
    self._nb_batch = int(np.ceil(self._num_instances/ float(self._batch_size)))
    self._shuffle_idx = np.arange(self._num_instances)
    self._batches = [(i*self._batch_size, min(self._num_instances, (i+1) * self._batch_size))
            for i in range(0, self._nb_batch)]

    # Queue holding batch ids
    self._batch_ids_queue = Queue(FLAGS.max_queue_size * 5)
    self._feed_queue = Queue(FLAGS.max_queue_size)
    self._data_status = DataStatus(self._batch_size, self._num_instances)
    self._data_status.reset()
    self._reset_batch_ptr = None
    self._reset_batches()


  def start(self):
    bi_threads = [mp.Process(target=self._fill_batch_ids_queue)]
    fd_threads = [mp.Process(target=self._fill_feed_queue, args=(i,))
                  for i in range(self._num_threads)]

    self._threads = bi_threads + fd_threads
    for t in self._threads:
      t.start()


  def terminate(self):
    for t in self._threads:
      t.terminate()

  def _fill_batch_ids_queue(self):
    while not self._coord.should_stop():
      ids = self._next_batch_ids()
      if ids is False:
        for thread_idx in range(self._num_threads):
          self._batch_ids_queue.put(False)
        break
      if len(ids) != self._batch_size and self._phase_train:
        continue
      self._batch_ids_queue.put(ids)

  def _fill_feed_queue(self, thread_idx):
    while not self._coord.should_stop():
      batch_ids = self._batch_ids_queue.get()
      if batch_ids is False:
        break
      data = self._get_data(batch_ids, thread_idx)
      self._feed_queue.put(data)

  def _next_batch_ids(self):
    self._batch_index += 1
    if self._batch_index == len(self._batches):
      if not self._phase_train:
        return False
      self._reset_batches()
      self._batch_index = 0

    batch_start, batch_end = self._batches[self._batch_index]
    return self._shuffle_idx[batch_start: batch_end]

  def _reset_batches(self):
    if self._phase_train:
      if self._reset_batch_ptr is None or self._reset_batch_ptr() is False:
        np.random.shuffle(self._shuffle_idx)

    self._batch_index = -1

  def next(self):
    self._data_status.update()
    try:
      batch = self._feed_queue.get(timeout=5)
    except:
      print("data loading finished")
      batch = False
    return batch

  def steps_per_epoch(self):
    return int(np.ceil(self._num_instances / float(self._batch_size)))

class DataStatus(object):
  def __init__(self, batch_size, n_samples):
    self._step = 0
    self._epoch = 0
    self._current_iter = 0
    self._batch_size = batch_size
    self._n_samples = n_samples

  def update(self):
    self._step += 1
    self._current_iter = min(self._step * self._batch_size, self._n_samples)
    if self._current_iter == self._n_samples:
      self._epoch += 1
      self._step = 0

  def reset(self):
    self._step = 0
    self._epoch = 0