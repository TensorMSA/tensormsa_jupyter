import tensorflow as tf
import numpy as np
import time
import math
import os
import sys

from lib import data_utils, model_utils
from configs import model_config


def main(_):
	config = model_config.Config()
	with tf.Session() as sess:
		forward_only = True

		vocab_path = os.path.join(config.data_dir, 'vocab%d.in' % config.input_vocab_size)

		# Load data
		vocab, vocab_rev = data_utils.load_vocabulary(vocab_path)

		config.batch_size = 1
		model = model_utils.create_model(sess, config, forward_only)


		valid_data_path = os.path.join(config.data_dir, 'chat_valid_ids%d.in'% config.input_vocab_size)
		dev_set = data_utils.read_test_data_chat(valid_data_path, config)[:2]
		bucket_id = 0

		for i in range(len(dev_set[0])):
			dev_inputs, dev_inputs_length, dev_outputs, dev_outputs_length, target_weights = (
			data_utils.get_test_line(dev_set[bucket_id], i))

			_, _, logits, predicted, enc_embedding, dec_embedding = model.step(sess, dev_inputs, dev_inputs_length,
				                                                        dev_outputs, dev_outputs_length, target_weights,forward_only)

			print("\nPrediction Results in Iteration %d : " % i)

			s = ""
			for input in dev_inputs:
				s += (vocab_rev[input[0]]) + " "
			print (s)

			s = ""
			for output in dev_outputs:
				s += (vocab_rev[output[0]]) + " "
			print (s)

			s = ""
			for i in predicted:
				s += (vocab_rev[i[0]]) + " "
			print (s)


if __name__ == '__main__':
  tf.app.run()
