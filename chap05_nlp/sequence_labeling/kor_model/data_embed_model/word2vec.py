from gensim.models import word2vec
from kor_model.data_embed_model.data_utils import write_vocab
import os
import numpy as np

def train_w2v(config) :
    """
    train word2vec model
    :param train_file:
    :return:
    """
    try :
        print("word2vec train start")
        update_flag = False
        model = word2vec.Word2Vec(size=300 , window=5, min_count=1, workers=4)

        with open(config.pos_path) as f :
            for line in f.readlines() :
                if (update_flag == False):
                    model.build_vocab([line.split(' ')], update=False)
                    update_flag = True
                else:
                    model.build_vocab([line.split(' ')], update=True)

        with open(config.pos_path) as f:
            for line in f.readlines():
                model.train(line.split(' '))

        os.makedirs(config.embedding_model_path, exist_ok=True)
        model.save(''.join([config.embedding_model_path, '/' , 'model']))
        return model

    except Exception as e :
        print (Exception("error on train w2v : {0}".format(e)))
    finally:
        print("word2vec train done")