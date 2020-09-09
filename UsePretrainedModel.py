# -*- coding: utf-8 -*-


import tensorflow as tf
import time
from model import Transformer
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)
logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
# load_hparams(hp, hp.ckpt)

input_tokens = tf.placeholder(tf.int32, shape=(1, None))
xs = (input_tokens, None, None)
logging.info("# Load model")
m = Transformer(hp)
y_hat = m.infer(xs)

logging.info("# Session")
sess = tf.Session()
# ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
# ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
saver = tf.train.Saver()
saver.restore(sess, './data/translation_model.ckpt')


def divide_long(text):
    return [i + '。' for i in text.split('。')]


def trans(text):
    # x = encode(text, "x", m.token2idx)
    tokens = [ch for ch in text] + ["</s>"]
    x = [m.token2idx.get(t, m.token2idx["<unk>"]) for t in tokens]
    pred = sess.run(y_hat, feed_dict={input_tokens: [x]})
    token_pred = [m.idx2token.get(t_id, "#") for t_id in pred[0]]
    translation = "".join(token_pred).split("</s>")[0]
    time.sleep(0.5)
    # print(translation)
    # logging.info("译文: " + translation)
    return translation


def main():
    text = input("请输入文言文：")
    if len(text) < 50:
        # print('Less than 50')
        return trans(text)
    else:
        trans_list = []
        text = divide_long(text)
        for i in text:
            trans_list.append(trans(i))
        trans_word = ''.join(trans_list)
        return trans_word


if __name__ == '__main__':
    main()
