import sys
import numpy as np
import random
import codecs
from configs import model_config


def load_vocabulary(vocab_path):
	rev_vocab = []
	#with open(vocab_path, mode="rU") as f:
	with codecs.open(vocab_path, mode="r",encoding="utf-8") as f:
		rev_vocab.extend(f.readlines())

	rev_vocab = [line.strip() for line in rev_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
	return vocab, rev_vocab


def read_data_chat(token_chat_path, config, max_size=None):
	data_set = [[] for _ in config.buckets]
	#print (data_set)

	with open(token_chat_path, mode="rU") as fh:
	#with codecs.open(token_chat_path, mode="r",encoding="utf-8") as fh:
		user_utt, bot_utt = fh.readline(), fh.readline()
		print (user_utt,bot_utt)
		counter = 0
		while user_utt and bot_utt and (not max_size or counter < max_size):
			counter += 1
			if counter % 1000 == 0:
				print("  reading data line %d" % counter)
				sys.stdout.flush()

			source_ids = [int(x) for x in user_utt.split()]
			for idx in range(len(source_ids)):
				if source_ids[idx] >= config.input_vocab_size:
					source_ids[idx] = model_config.UNK_ID

			target_ids = [int(x) for x in bot_utt.split()]
			for idx in range(len(target_ids)):
				if target_ids[idx] >= config.target_vocab_size:
					target_ids[idx] = model_config.UNK_ID

			for bucket_id, (source_size, target_size) in enumerate(config.buckets):
				if len(source_ids) <= source_size and len(target_ids) < target_size:
					data_set[bucket_id].append([source_ids, target_ids])
					break
			user_utt, bot_utt =  fh.readline(), fh.readline()
		return data_set



def read_test_data_chat(token_chat_path, config, max_size=None):
	data_set = [[] for _ in config.buckets]
	#print (data_set)

	with open(token_chat_path, mode="rU") as fh:
	#with codecs.open(token_chat_path, mode="r",encoding="utf-8") as fh:
		user_utt, bot_utt = fh.readline(), fh.readline()
		print (user_utt,bot_utt)
		counter = 0
		while user_utt and bot_utt and (not max_size or counter < max_size):
			counter += 1
			if counter % 1000 == 0:
				print("  reading data line %d" % counter)
				sys.stdout.flush()

			source_ids = [int(x) for x in user_utt.split()]
			for idx in range(len(source_ids)):
				if source_ids[idx] >= config.input_vocab_size:
					source_ids[idx] = model_config.UNK_ID

			target_ids = [int(x) for x in bot_utt.split()]
			for idx in range(len(target_ids)):
				if target_ids[idx] >= config.target_vocab_size:
					target_ids[idx] = model_config.UNK_ID

			data_set[0].append([source_ids, target_ids])

			user_utt, bot_utt =  fh.readline(), fh.readline()
		return data_set


def get_batch(chat_token_ids, config):
	rnd_idx = np.random.randint(0, len(chat_token_ids), config.batch_size)
	# print("data_utils.get_batch")
	# print(rnd_idx)
	# rnd_idx = [0,1,2,3]
	encoder_inputs = []
	decoder_inputs = []
	encoder_inputs_length = []
	decoder_inputs_length = []
	target_weights = []
	for i in range(config.batch_size):
		encoder_inputs.append(chat_token_ids[rnd_idx[i]][0])
		decoder_inputs.append(chat_token_ids[rnd_idx[i]][1])
		encoder_inputs_length.append(len(chat_token_ids[rnd_idx[i]][0]))
		decoder_inputs_length.append(len(chat_token_ids[rnd_idx[i]][1]))

	max_encoder_length = max(encoder_inputs_length)
	max_decoder_length = max(decoder_inputs_length)
	for i in range(config.batch_size):
		temp_encoder = encoder_inputs[i]
		# for j in range(max_encoder_length - encoder_inputs_length[i]):
		# 	temp_encoder.append(model_config.PAD_ID)
		encoder_pad = [model_config.PAD_ID] * (max_encoder_length - encoder_inputs_length[i])
		encoder_inputs[i] = list(temp_encoder + encoder_pad)

		temp_decoder = decoder_inputs[i]
		decoder_pad = [model_config.PAD_ID] * (max_decoder_length - decoder_inputs_length[i])
		# for j in range(max_decoder_length - decoder_inputs_length[i]):
		# 	temp_decoder.append(model_config.PAD_ID)
		decoder_inputs[i] = list([model_config.GO_ID] + temp_decoder + decoder_pad)
		weight = list([1] * decoder_inputs_length[i] + [0] * (max_decoder_length - decoder_inputs_length[i]))
		target_weights.append(weight)

	encoder_inputs2 = np.asarray(encoder_inputs).transpose()
	encoder_inputs_length2 = np.asarray(encoder_inputs_length)
	decoder_inputs2 = np.asarray(decoder_inputs).transpose()
	decoder_inputs_length2 = np.asarray(decoder_inputs_length)
	target_weights2 = np.asarray(target_weights)

	return encoder_inputs2, encoder_inputs_length2, decoder_inputs2, decoder_inputs_length2, target_weights2


def get_test_line(chat_token_ids, idx):
	encoder_inputs = [chat_token_ids[idx][0]]
	encoder_inputs_length = [len(chat_token_ids[idx][0])]
	decoder_inputs = [list([model_config.GO_ID] + chat_token_ids[idx][1])]
	decoder_inputs_length = [len(chat_token_ids[idx][1])]
	encoder_inputs2 = np.asarray(encoder_inputs).transpose()
	encoder_inputs_length2 = np.asarray(encoder_inputs_length)
	decoder_inputs2 = np.asarray(decoder_inputs).transpose()
	decoder_inputs_length2 = np.asarray(decoder_inputs_length)
	target_weights = [[1] * len(chat_token_ids[idx][1])]
	return encoder_inputs2, encoder_inputs_length2, decoder_inputs2, decoder_inputs_length2, target_weights


def get_batch2(chat_token_ids, config):
	random_data = multi_random(chat_token_ids, config.batch_size)
	encoder_inputs_length = []
	decoder_inputs_length = []
	for i in range(config.batch_size):
		encoder_inputs_length.append(len(random_data[i][0]))
		decoder_inputs_length.append(len(random_data[i][1]))

	max_encoder_length = max(encoder_inputs_length)
	max_decoder_length = max(decoder_inputs_length)
	encoder_inputs = []
	decoder_inputs = []
	for i in range(config.batch_size):
		encoder_pad = [model_config.PAD_ID] * (max_encoder_length - encoder_inputs_length[i])
		decoder_pad = [model_config.PAD_ID] * (max_decoder_length - decoder_inputs_length[i])
		encoder_inputs.append(list(random_data[i][0] + encoder_pad))
		decoder_inputs.append(list(random_data[i][1] + decoder_pad))

	encoder_inputs2 = np.asarray(encoder_inputs).transpose()
	encoder_inputs_length2 = np.asarray(encoder_inputs_length)
	decoder_inputs2 = np.asarray(decoder_inputs).transpose()
	decoder_inputs_length2 = np.asarray(decoder_inputs_length)

	return encoder_inputs2, encoder_inputs_length2, decoder_inputs2, decoder_inputs_length2


def multi_random(data_array, n):
	result_array = []
	for i in range(n):
		result_array.append(random.choice(data_array))
	return result_array




def read_data(tokenized_dialog_path, max_size=None):
  """Read data from source file and put into buckets.

  Args:
    source_path: path to the files with token-ids.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in BUCKETS]

  with gfile.GFile(tokenized_dialog_path, mode="r") as fh:
      source, target = fh.readline(), fh.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          #print("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = fh.readline(), fh.readline()
  return data_set