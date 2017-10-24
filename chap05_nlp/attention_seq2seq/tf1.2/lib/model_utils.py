import tensorflow as tf

from tensorflow.python.platform import gfile
from lib import chat_seq2seq_model


def create_model(session, config, forward_only):
	model = chat_seq2seq_model.ChatSeq2SeqModel(
		config=config,
		use_lstm=True,
		forward_only=forward_only,
		attention=True
	)

	ckpt = tf.train.get_checkpoint_state(config.model_dir)
	if ckpt and gfile.Exists("%s.index" % ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model
