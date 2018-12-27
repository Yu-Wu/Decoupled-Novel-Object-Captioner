
import tensorflow as tf
from utils.models_base import NNModel
from lib.rnn import rnn
from lib.rnn import rnn_cell
from lib.rnn import seq2seq
import numpy as np
import pickle as pkl
import random, time

FLAGS = tf.app.flags.FLAGS

class DNOC(NNModel):
    def _get_dec_cell(self, cell_size):
        single_cell = rnn_cell.BasicLSTMCell(num_units=cell_size, state_is_tuple=False)
        if self._phase_train:
            single_cell = rnn_cell.DropoutWrapper(
                    single_cell, output_keep_prob=0.5, input_keep_prob=0.5)

        cell = rnn_cell.OutputProjectionWrapper(single_cell, self._target_vocab_size)
        return cell

    def __init__(self, num_total_vocab, num_lstm_vocabs):
        super(DNOC, self).__init__()
        self._batch_size = FLAGS.batch_size
        self._dec_seq_len = FLAGS.dec_seq_len
        self._cnn_fea_dim = FLAGS.cnn_fea_dim
        self._dec_cell_size = FLAGS.dec_cell_size
        self._enc_cell_size = FLAGS.enc_cell_size
        self.num_lstm_vocabs = num_lstm_vocabs
        self._word_embedding_size = FLAGS.word_embedding_size

        # vocab size for the SM-P output
        self._target_vocab_size = num_lstm_vocabs + 1  # We define the last class of LSTM words to be <PL>.
        self._phase_train = FLAGS.stage == "train"

        self._caption_outs = {}
        self._dec_inputs = []
        self.target_weights = []

        self.dec_cell = self._get_dec_cell(self._dec_cell_size)
        
        # Inputs
        self._enc_inputs = tf.placeholder(
                    tf.float32,
                    shape=[None, self._cnn_fea_dim],
                    name="encode")

        for i in range(self._dec_seq_len):
            self._dec_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        self._det_feature = tf.placeholder(tf.float32, shape=[self._batch_size, FLAGS.max_det_boxes, FLAGS.det_fea_dim], name="det_feature")
        self._det_value = tf.placeholder(tf.int32, [self._batch_size, FLAGS.max_det_boxes], name="det_value")
        

        # Targets
        targets = [self._dec_inputs[i + 1]
                             for i in range(len(self._dec_inputs) - 1)]
        targets += [tf.zeros_like(self._dec_inputs[0])]
        self._query_targets = [] # the query target ground truth
        self._query_target_weights = [] # defines wthether the query loss is calculated at this pos
        for i in range(self._dec_seq_len):
            self._query_targets.append(tf.placeholder(tf.int32, shape=[None], name="query_target{0}".format(i)))
            self._query_target_weights.append(tf.placeholder(tf.float32, shape=[None], name="query_target_weight{0}".format(i)))

        # Encoder.
        image_feature = self._enc_inputs 
        if self._phase_train:
            image_feature = tf.nn.dropout(image_feature, keep_prob=0.5)
        image_feature = tf.contrib.layers.fully_connected(image_feature, FLAGS.dec_cell_size * 2, activation_fn=None, scope="image_feature")
        self._enc_state = image_feature                  # image feature

        # Decoder.
        output_projection = None
        output_size = self._target_vocab_size

        # Lstm output
        self.outputs, hidden_states = seq2seq.embedding_rnn_decoder(
                        self._dec_inputs, self._enc_state, self.dec_cell,
                        self._target_vocab_size, self._word_embedding_size,
                        output_projection=output_projection,
                        feed_previous=(not self._phase_train))

        # Query outputs
        self.query_outputs = self.query_memory(hidden_states)

        # Calculate loss
        if self._phase_train:
            self._lstm_loss = seq2seq.sequence_loss(
                    self.outputs, targets, self.target_weights,
                    softmax_loss_function=None)
            self._query_loss = seq2seq.sequence_loss(
                    self.query_outputs, self._query_targets, self._query_target_weights, average_across_timesteps=True,
                    softmax_loss_function=None)
            self._total_loss = (self._lstm_loss + self._query_loss)/2.0
            self._get_train_op(clip_norm='global')


    def query_memory(self, hidden_states):
        # prepare data
        def prepare_query_feature_values():
            query_feature = self._det_feature
            if self._phase_train:
                query_feature =  tf.nn.dropout(query_feature, keep_prob=0.5)
            query_value = tf.one_hot(self._det_value, FLAGS.detection_classes)
            return query_feature, query_value

        # key value attention 
        def key_value_att(att_query, att_key, att_value):   
            with tf.variable_scope("Attention"):
                weights = tf.matmul(att_key, att_query)
                weights = tf.transpose(weights, perm=[0, 2, 1])
                att_result = tf.matmul(weights, att_value)
                att_result = tf.reshape(att_result, [att_result.shape[0], att_result.shape[2]])
            return att_result

        with tf.variable_scope("generate_query_word", reuse=False):
            query_outputs = []
            for i, hstate in enumerate(hidden_states):
                if i > 0:
                        tf.get_variable_scope().reuse_variables()

                # gen attation query, feature, value 
                att_query = tf.split(hstate, 2, 1)[1] # choose hidden states as the query
                att_query = tf.expand_dims(att_query, 2)

                att_key, att_value = prepare_query_feature_values()
                att_key = tf.contrib.layers.fully_connected(att_key, FLAGS.dec_cell_size, scope="att_key")

                if self._phase_train:
                    att_query = tf.nn.dropout(att_query, keep_prob=0.5)
                    att_key = tf.nn.dropout(att_key, keep_prob=0.5)                    

                # query the memory and output the novel words logits 
                query_predict = key_value_att(att_query=att_query, att_key=att_key, att_value=att_value)
                query_outputs.append(query_predict)
        return query_outputs



    def _get_input_feed_dict(self, batch):
        input_feed = {}
        input_feed[self._enc_inputs.name] = batch['inputs']
        input_feed[self._det_feature.name] = batch['inputs_det_feats']
        input_feed[self._det_value.name] = batch['inputs_det_values']

        if self._phase_train:
            _dec_inputs = batch['decoder_inputs']
            target_weights = batch['dec_target_weights']   
            query_targets = batch["query_targets"]
            query_target_weights = batch["query_target_weights"]
        else: # for test
            bs = len(batch['inputs'])
            _dec_inputs, target_weights = [], []
            _dec_inputs.append([FLAGS.GO_ID for _ in range(bs)])
            target_weights.append([0.])
            for i in range(self._dec_seq_len - 1):
                _dec_inputs.append([FLAGS.PAD_ID for _ in range(bs)])
                target_weights.append([0.])

            query_targets, query_target_weights = [], []
            for l in range(self._dec_seq_len):
                query_targets.append([0])
                query_target_weights.append([0.])

        for l in range(self._dec_seq_len):
            input_feed[self._dec_inputs[l].name] = _dec_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self._query_targets[l].name] = query_targets[l]
            input_feed[self._query_target_weights[l].name] = query_target_weights[l]
        return input_feed

    def train_step(self, sess, batch):
        input_feed = self._get_input_feed_dict(batch)
        output_feed = [self._train_op, self._total_loss, self._query_loss]
        loss = sess.run(output_feed, input_feed)
        return loss[1:]

    def eval_step(self, sess, batch):
        input_feed = self._get_input_feed_dict(batch)
        # output feed: [0:self._dec_seq_len] is the lstm output
        # output feed: [self._dec_seq_len:] is the query output
        output_feed = []        
        for l in range(self._dec_seq_len):  # Output logits.
            output_feed.append(self.outputs[l])
        for l in range(self._dec_seq_len):  # Output query logits.
            output_feed.append(self.query_outputs[l])
        # run the model
        outputs = sess.run(output_feed, input_feed)
        lstm_outputs = outputs[:self._dec_seq_len]
        query_outputs =  outputs[self._dec_seq_len:]

        # get predictions from logits with maximum values
        lstm_outputs = np.argmax(np.transpose(np.array(lstm_outputs), (1, 0, 2)), axis=2)
        query_outputs = np.argmax(np.transpose(np.array(query_outputs), (1, 0, 2)), axis=2)
         
        # We define the last class of LSTM words to be <PL>.
        # Therefore, if the lstm output prediction equals "self.num_lstm_vocabs",
        # we should take the word from query results to feed in the <PL>. 
        query_outputs[lstm_outputs != self.num_lstm_vocabs] = 0

        # In generating the held-out coco dataset, we reorganzied the vocab sequence,
        # so that detection words are at the last 80 places in the idx-to-word list.
        # For those need to be replaced, the lstm_output is self.num_lstm_vocabs,
        # the query_outputs (in [0, 80]) is the indexs of detection words. 
        # Therefore, lstm_outputs + query_outputs is the index for the final output.
        final_outputs = lstm_outputs + query_outputs

        # replace idx to word
        final_outputs = np.vectorize(batch['vocab'].get)(final_outputs)
        captions = []
        for idx in range(len(batch['inputs'])):
            sent = list(final_outputs[idx]) + ["<eos>"]
            sent = sent[:sent.index("<eos>")]
            captions.append(sent)
        return captions
