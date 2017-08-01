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
		forward_only = False

		vocab_path = os.path.join(config.data_dir, 'vocab%d.in' % config.input_vocab_size)

		train_data_path = os.path.join(config.data_dir, 'chat_ids%d.in' % config.input_vocab_size)

		# Load data
		vocab, vocab_rev = data_utils.load_vocabulary(vocab_path)
		train_set = data_utils.read_data_chat(train_data_path, config)
		# print(train_set[0])

		if forward_only:
			config.batch_size = 1
			model = model_utils.create_model(sess, config, forward_only)
		else:
			model = model_utils.create_model(sess, config, forward_only)

		# This is the training loop.
		steps_per_checkpoint = 100
		step_time, loss = 0.0, 0.0
		current_step = 0
		perplexity = 10000.0
		previous_losses = []

		while current_step < config.max_epoch and not forward_only:
			start_time = time.time()
			bucket_id = 0
			encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, target_weights = (
				data_utils.get_batch(train_set[bucket_id], config))

			_, step_loss, _, _, enc_embedding, dec_embedding = model.step(sess, encoder_inputs, encoder_inputs_length,
			                                                              decoder_inputs, decoder_inputs_length, target_weights,forward_only)

			step_time += (time.time() - start_time) / 100
			loss += step_loss / 100
			current_step += 1

			if current_step % 100 == 0:
				# Print statistics for the previous epoch.
				# loss *= config.max_state_length 		# Temporary purpose only
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print("global step %d learning rate %.4f step-time %.2f perplexity %.2f loss %.2f" %
							(model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity, loss))

				if len(previous_losses) > 2 and loss > max(previous_losses[-2:]):
					# if len(previous_losses) > 0 and loss > previous_losses[-1:]:
					sess.run(model.learning_rate_decay_op)

				previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(config.model_dir, "model.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0

				sys.stdout.flush()

		if forward_only:
			valid_data_path = os.path.join(config.data_dir, 'chat_valid_ids%d.in'% config.input_vocab_size)
			dev_set = data_utils.read_data_chat(valid_data_path, config)
			print (dev_set)
			bucket_id = 0
			# for i in range(len(dev_set[0])):
			for i in range(1):
				dev_inputs, dev_inputs_length, dev_outputs, dev_outputs_length, target_weights = (
					data_utils.get_test_line(train_set[bucket_id], i))

				_, _, logits, predicted, enc_embedding, dec_embedding = model.step(sess, dev_inputs, dev_inputs_length,
				                                                        dev_outputs, dev_outputs_length, target_weights,forward_only)

				print("Prediction Results in Iteration %d : " % i)
				print(dev_inputs.transpose())
				print(dev_outputs.transpose())
				print(predicted.transpose())
				print("")


if __name__ == '__main__':
  tf.app.run()
