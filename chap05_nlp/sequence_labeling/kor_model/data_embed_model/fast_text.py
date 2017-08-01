import gensim, os
import fasttext
from gensim.models.wrappers import fasttext as gft

from gensim.models import word2vec
from kor_model.config import config

def load_pretrained_fasttext() :
    # Set FastText home to the path to the FastText executable
    ft_home = '/home/dev/fastText/fasttext'

    # Set file names for train and test data
    train_file = config.pos_path

    try :
        # Use FaceBook Corpus
        print(help(fasttext))
        fasttext.cbow('/home/dev/tensormsa_jupyter/chap05_nlp/wordembedding/data/test3.txt', 'model')
        model = gft.FastText.load_fasttext_format('/home/dev/tensormsa_jupyter/chap05_nlp/wordembedding/data/test3.model')
        result = model.most_similar(positive=['마법'])
        return result
    except Exception as e :
       raise Exception (e)

load_pretrained_fasttext()