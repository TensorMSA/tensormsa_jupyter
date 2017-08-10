import os
import tensorflow as tf
from config import Config
from data_handler import read_data, save_obj
from model import MemN2N

def run(is_test = False):
    count = []
    word2idx = {}
    Config.is_test = is_test
    if not os.path.exists(Config.checkpoint_dir):
      os.makedirs(Config.checkpoint_dir)
    if not os.path.exists(Config.vector_dir):
      os.makedirs(Config.vector_dir)

    train_data = read_data('%s/%s.train.txt' % (Config.data_dir, Config.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (Config.data_dir, Config.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (Config.data_dir, Config.data_name), count, word2idx)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    save_obj('%s/idx2word.pkl' % (Config.vector_dir), idx2word)
    save_obj('%s/word2idx.pkl' % (Config.vector_dir), word2idx)
    Config.nwords = len(word2idx)

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MemN2N(Config, sess, True)
        model.build_model()

        if Config.is_test:
            model.run(valid_data, test_data)
        else:
            model.run(train_data, valid_data)

        tf.summary.FileWriter("./logs", graph=tf.get_default_graph())


