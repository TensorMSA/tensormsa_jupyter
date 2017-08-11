import train, predict
import tensorflow as tf

def main(_) :
    # # train
    # train.run(False)
    #
    # # eval
    # train.run(True)

    # predict
    test_context = "but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of" \
                   " it in the final hour it barely managed to stay this side of chaos some circuit breakers installed after the october N crash " \
                   "failed their first test traders say unable to cool the selling panic in both stocks and futures the N stock specialist firms on " \
                   "the big board floor the buyers and sellers of last resort who were criticized after the N crash once again could n't handle the " \
                   "selling pressure big investment banks refused to step up to the plate to support the beleaguered floor traders by buying big blocks" \
                   " of stock traders say heavy selling of standard & poor 's 500-stock index futures in chicago <unk> beat stocks downward "
    test_question = "how was the test on october?"
    predict.run(test_context, test_question)

if __name__ == '__main__':
    tf.app.run()