import json
import numpy as np
import os, math, time
import tensorflow as tf

import multiprocessing as mp
from multiprocessing import Queue

from utils.dataloader import DataLoader

FLAGS = tf.app.flags.FLAGS


class VisualTextData(DataLoader):
  def __init__(self, coord):
    super(VisualTextData, self).__init__(coord)
    self._dec_seq_len = FLAGS.dec_seq_len
    self._cnn_fea_dim = FLAGS.cnn_fea_dim

    self.PAD_ID = FLAGS.PAD_ID
    self.GO_ID = FLAGS.GO_ID
    self.EOS_ID = FLAGS.EOS_ID
    self.UNK_ID = FLAGS.UNK_ID

    self._query_vocab = self._dataset["det_vocabs"]
    self._lstm_vocab = self._dataset["lstm_vocabs"]
    self._total_vocab = self._lstm_vocab + self._query_vocab
    self._wordtoix = self._dataset["word_to_ix"]
    self._ixtoword = self._dataset['ix_to_word']

    self.det_dir = FLAGS.det_dir
    self._lstm_vocab_num = len(self._lstm_vocab)
    self.detection_class_to_idx = {}
    d = self._dataset['detection_class_to_idx']
    for k in d:
      self.detection_class_to_idx[int(k)] = d[k]
      
  def _set_num_instances(self):
    self._text_data = self._dataset[FLAGS.stage]
    self.names = sorted(list(set([d['vname'] for d in self._text_data])))
    self._num_instances = len(self._text_data) if self._phase_train else len(self.names)


  def _get_data_train(self, idxs, thread_idx):
    vnames = []
    captions = []
    bs = len(idxs)
    for batch_idx in idxs:
      instance = self._text_data[batch_idx]
      vnames.append(instance['vname'])
      caption = instance['caption_inputs']
      if len(caption) > self._dec_seq_len - 2:
        caption = [self.GO_ID] + caption[: self._dec_seq_len - 2] + [self.EOS_ID]
      else:
        caption = [self.GO_ID] + caption + [self.EOS_ID]
      dec_pad_size = self._dec_seq_len - len(caption)
      captions.append(caption + [self.PAD_ID] * dec_pad_size)

    # batch x seq x pool x fea_vec
    input_feats = self.get_image_feature(vnames)
    det_feats, det_values = self.get_support_feat(vnames)

    dec_inputs = []
    dec_target_weights = []
    query_targets = []
    query_target_weights = []
    for seq_id in range(self._dec_seq_len):
      dec_input = []
      query_target = []
      dec_target_weight = np.ones(bs, dtype=np.float32)
      query_target_weight = np.zeros(bs, dtype=np.float32)
      for batch_id in range(bs):
    
	# To transfer detection words to <PL>  
        _dec_input = min(captions[batch_id][seq_id], self._lstm_vocab_num)
        dec_input.append(_dec_input)

        if seq_id < self._dec_seq_len - 1:
          dec_target = captions[batch_id][seq_id + 1]

        if seq_id == self._dec_seq_len - 1 or dec_target == self.PAD_ID:
          dec_target_weight[batch_id] = 0.0

        # find out if the targe word is query
        _query_target = 0
        if seq_id < self._dec_seq_len - 1:
          target = captions[batch_id][seq_id + 1]
          if target >= self._lstm_vocab_num:
            _query_target = target - self._lstm_vocab_num
            query_target_weight[batch_id] = 1.0
        query_target.append(_query_target)

      dec_inputs.append(dec_input)
      query_targets.append(query_target) 
      dec_target_weights.append(dec_target_weight)
      query_target_weights.append(query_target_weight)

    outputs = {}
    outputs['inputs'] = input_feats
    outputs['names'] = vnames
    outputs['decoder_inputs'] = dec_inputs
    outputs['dec_target_weights'] = dec_target_weights
    outputs["inputs_det_feats"] = det_feats
    outputs["inputs_det_values"] = det_values
    outputs['query_targets'] = query_targets
    outputs['query_target_weights'] = query_target_weights

    return outputs


  def _get_data_eval(self, idxs, thread_idx):
    #assert len(idxs) == 1
    bs = len(idxs)
    vnames = [self.names[batch_idx] for batch_idx in idxs]
    if bs != FLAGS.batch_size:
      vnames += [vnames[-1]] * (FLAGS.batch_size - bs)

    input_feats = self.get_image_feature(vnames)
    det_feats, det_values = self.get_support_feat(vnames)
    
    outputs = {}
    outputs["inputs_det_feats"] = det_feats
    outputs["inputs_det_values"] = det_values
    outputs['inputs'] = input_feats
    outputs['names'] = vnames
    outputs['vocab'] = {}
    for idx, o in enumerate(self._total_vocab):
      outputs['vocab'][idx] = o

    return outputs


  def get_image_feature(self, names):
    bs = len(names)
    feats = []
    for batch_idx, name in enumerate(names):
      path = os.path.join(FLAGS.cnn_dir, name) + ".npz"
      feats.append(np.load(path)['cnn_feat'])
    feats = np.stack(feats, axis=0)
    return feats


  def get_support_feat(self, vnames, return_array=True):
    bs = len(vnames)
    det_feats=np.zeros([bs, FLAGS.max_det_boxes, FLAGS.det_fea_dim])
    det_values=np.zeros([bs, FLAGS.max_det_boxes])

    for batch_idx, vname in enumerate(vnames):
      path = os.path.join(FLAGS.det_dir, vname) + ".npz"
      det_np_file = np.load(path) 
      det_feat = det_np_file['det_feature']
      det_value = det_np_file['det_classes'].astype(int)

      # Map the detection output class to consecutive int numbers
      det_value = np.vectorize(self.detection_class_to_idx.get)(det_value)
      length = det_value.shape[0]
      max_length = min(length, FLAGS.max_det_boxes)
      if max_length == 0: # if no boxes in the image
        continue
      det_feats[batch_idx][:max_length] = det_feat[:max_length]
      det_values[batch_idx][:max_length] = det_value[:max_length]
    return det_feats, det_values

  def _get_data(self, batch_idxs, thread_idx):
    if self._phase_train:
      return self._get_data_train(batch_idxs, thread_idx)
    else:
      return self._get_data_eval(batch_idxs, thread_idx)


  def get_vocab(self):
    return len(self._total_vocab), len(self._lstm_vocab)


  def get_eval_score(self, pred_captions):
    print("Begin to calculate evaluation scores...(It may takes a few minutes)")
        
    import utils.captioning_metrics as metric
    gt, res = metric.load_captions(pred_captions, self._text_data)
    assert len(gt.keys()) == len(res.keys()), "Error gt {}  res {}".format(len(gt.keys()), len(res.keys()))
    ids = gt.keys()
    metric.score(gt, res, ids)
