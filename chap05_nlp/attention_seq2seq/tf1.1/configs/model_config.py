import tensorflow as tf

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

class Config():
	root_dir = './'
	data_dir = root_dir+'data'
	model_dir = root_dir+'nn_models'
	results_dir = root_dir+'results'
	reply_dir = root_dir+'reply'

	data_type = tf.float32

	learning_rate = 0.05
	learning_rate_decay_factor = 0.99
	max_gradient_norm = 5.0
	# keep_prob = 1.0

	input_vocab_size = 8000
	target_vocab_size = 426
	batch_size = 4
	enc_hidden_size = 128
	enc_num_layers = 1
	dec_hidden_size = 128
	dec_num_layers = 1

	max_epoch = 100000
	checkpoint_step = 100

	buckets = [(8, 15)]
