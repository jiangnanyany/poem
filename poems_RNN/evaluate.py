import tensorflow as tf 
import numpy as np 
import utils
from model import RNN_poem
data = utils.Dataset('./data/poems_src.txt')
word2index = data.word2index
vocab_size = len(word2index)
mode = 'b' # 'a' 自动生成  'b'藏头诗
print
with tf.Session() as sess:
    model = RNN_poem(vocab_size, 100, 100, test=False)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, tf.train.latest_checkpoint('save_model'))
    if mode == 'a':
        predict_word = [word2index[utils.start_token]]
        poem = []
        last_state = np.zeros([1, 2, 100], dtype=np.float32)
        for i in range(64):
            state, predict_word = sess.run([model.state, model.predict_word], 
                       feed_dict={model.last_word: predict_word, model.last_state: last_state})
            c, h = state[0], state[1]
            last_state = np.stack([c, h], axis=1)
            word = data.index2word[predict_word[0]]
            if word == utils.end_token:
                break
            poem.append(word)
        print(''.join(poem))
    
    if mode == 'b':
        heads = '墨雨南怜'
        j = 0
        predict_word = [word2index[utils.start_token]]
        poem = []
        last_state = np.zeros([1, 2, 100], dtype=np.float32)
        for i in range(64):
            flag=False
            if (i==0) or (word=='，') or (word == '。'):
                flag=True
            state, predict_word = sess.run([model.state, model.predict_word], 
                       feed_dict={model.last_word: predict_word, model.last_state: last_state})
            c, h = state[0], state[1]
            last_state = np.stack([c, h], axis=1)
            word = data.index2word[predict_word[0]]
            if word == utils.end_token:
                break
            if flag == True:
                if j >= len(heads):
                    break
                word = heads[j]
                j = j + 1
                predict_word = [word2index[word]]
            poem.append(word)
        print(''.join(poem))