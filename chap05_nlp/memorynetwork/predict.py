import os
import tensorflow as tf
import numpy as np
from config import Config
from data_handler import read_data,load_obj, read_txt
from model import MemN2N


def run(context, question):
    word2idx = {}
    idx2word = {}

    idx2word = load_obj('%s/idx2word.pkl' % (Config.vector_dir), idx2word)
    word2idx = load_obj('%s/word2idx.pkl' % (Config.vector_dir), word2idx)
    context_data = read_txt(context, word2idx)
    question_data = read_txt(question, word2idx)
    Config.nwords = len(word2idx)

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MemN2N(Config, sess, False)
        model.build_model()
        results = model.predict(context_data, question_data)
        for result in results :
            print(' '.join(list(map(lambda x : idx2word.get(np.argmax(x)) , result[0]))))