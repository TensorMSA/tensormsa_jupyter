from kor_model.data_crawler import crawler
from kor_model.data_crawler import mecab
from kor_model.data_embed_model import build_data
from kor_model.config import config
from kor_model.ner_model.lstmcrf_model import NERModel
from kor_model.general_utils import get_logger
from kor_model.data_embed_model import data_utils
from kor_model.data_embed_model.data_utils import CoNLLDataset
import os

# (3) train NER Model (1.bilstm-crf, 2.attention)
# get data
embeddings = data_utils.get_trimmed_glove_vectors(config.trimmed_filename)
char_embedding = data_utils.get_trimmed_glove_vectors(config.charembed_filename)
vocab_words = data_utils.load_vocab(config.words_filename)
vocab_tags = data_utils.load_vocab(config.tags_filename)
vocab_chars = data_utils.load_vocab(config.chars_filename)

processing_word = data_utils.get_processing_word(vocab_words,
                                                 vocab_chars,
                                                 lowercase=config.lowercase,
                                                 chars=config.chars)
processing_tag = data_utils.get_processing_word(vocab_tags,
                                                lowercase=False)

# vocab_chars = data_utils.load_vocab(config.chars_filename)
# create dataset
dev = CoNLLDataset(config.dev_filename, processing_word, processing_tag, config.max_iter)
test = CoNLLDataset(config.test_filename, processing_word, processing_tag, config.max_iter)
train = CoNLLDataset(config.train_filename, processing_word, processing_tag, config.max_iter)

# build model
model = NERModel(config, embeddings, ntags=len(vocab_tags),nchars=len(vocab_chars), logger=None, char_embed=char_embedding)
model.build()
model.train(train, dev, vocab_tags)
model.evaluate(test, vocab_tags)