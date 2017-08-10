from collections import namedtuple
from gensim.models import Doc2Vec
from konlpy.tag import Mecab
from konlpy.tag import Twitter
from pprint import pprint
import scipy.spatial as spatial
import numpy as np
import sys
import codecs

TaggedDocument = namedtuple('TaggedDocument', 'words tags')

# windows10에서는 konlpy에서 mecab지원하지 않음
def tokenize_mecab(doc):
    pos_tagger = Mecab()
    return ['/'.join(t) for t in pos_tagger.pos(doc)]


def tokenize_twitter(doc):
    pos_tagger = Twitter()
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


def read_train_data(filename):
    with codecs.open(filename, 'r', "utf-8") as f:
        data = [line.rstrip('\n') for line in f]
    return data


def avg_feature_vector(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def main(mode):
    if('train' == mode):
        print(u'\nData Loading...', end=' ')
        train_data = read_train_data('train_findpeople.txt')
        print(u'[OK]')

        print(u'Tokenizing...', end=' ')
        train_docs = [(tokenize_twitter(row), 1) for row in train_data]
        print(train_docs)
        print(u'[OK]')

        print(u'Document Tagging...', end=' ')
        tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
        print(u'[OK]')

        print(u'Build vocab...', end=' ')
        doc_vectorizer = Doc2Vec(dm=0, window=15, size=300, alpha=0.025, min_alpha=0.025, seed=9999, min_count=1)
        doc_vectorizer.build_vocab(tagged_train_docs)
        print(u'[OK]')

        #print(doc_vectorizer.corpus_count, doc_vectorizer.iter)
        print(u'Doc2Vec Training...', end=' ')
        for epoch in range(10):
            # For gensim 1.0.1
            # doc_vectorizer.train(tagged_train_docs)
            # For gensim 2.0.0
            doc_vectorizer.train(tagged_train_docs, total_words=None, word_count=0, total_examples=doc_vectorizer.corpus_count, queue_factor=2, report_delay=1.0, epochs=doc_vectorizer.iter)

            doc_vectorizer.alpha -= 0.002  # decrease the learning rate
            doc_vectorizer.min_alpha = doc_vectorizer.alpha
        print(u'[OK]')

        print(u'Save doc2vec model file...', end=' ');
        doc_vectorizer.save('doc2vec.model')
        print(u'[OK]')

        print(u'\nTraining Complete!');
    elif('test' == mode):
        print(u'\nDoc2vec model loading...', end=' ');
        ref_model = Doc2Vec.load('doc2vec.model')
        print(u'[OK]')

        print(u'Index2word : {')
        print(ref_model.wv.index2word)
        print(u'}')

        sentence = input(u'Input sentence : ')
        sentence_avg_vector = avg_feature_vector(tokenize_twitter(sentence), model=ref_model, num_features=300)

        avg_val = 0.
        ref_text = ''

        ############################
        # Intent Analysis Testing...
        train_file = open('train_findpeople.txt', 'rt', encoding='UTF8')
        try:
            for line in train_file:
                ref_sentence_avg_vector = avg_feature_vector(tokenize_twitter(line), model=ref_model, num_features=300)
                similarity_comparison_val = 1 - spatial.distance.cosine(sentence_avg_vector, ref_sentence_avg_vector)
                if(similarity_comparison_val > avg_val):
                    avg_val = similarity_comparison_val
                    ref_text = line

            print('TOP :', avg_val, '(' + ref_text.splitlines()[0] + '), ', end=' ')
        finally:
            train_file.close()

        ###########################
        # NER Testing...
        entity_file = open('entity_findpeople.txt', 'rt', encoding='UTF8')
        try:
            for line in entity_file:
                for item in line.split(','):
                    item = item.strip()
                    if(sentence.count(item) > 0):
                        sentence = sentence.replace(item, '{'+line.split(',')[0] +'}')

            print('NER : ' + sentence)
        finally:
            entity_file.close()
    else:
        print(u'Please, select train or test mode.')


if __name__ == "__main__" :
    main(sys.argv[1])
