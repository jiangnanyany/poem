import tensorflow as tf 
import utils
from model import RNN_poem
import numpy as np
import os

epochs = 50
data = utils.Dataset('./data/poems_src.txt')
word2index = data.word2index
vocab_size = len(word2index)

if not os.path.exists('save_model'):
    os.makedirs('save_model')


with tf.Session() as sess:
    model = RNN_poem(vocab_size, 100, 100, test=False)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    for i in range(epochs):
        losses = []
        for x_data, y_data in data.get_batch(32):
            feeds = {
                model.input_text: x_data,
                model.target_text: y_data
            }
            _, loss = sess.run([model.train_op, model.loss], feeds)
            losses.append(loss)
        print(np.mean(losses))
        tensorflow_file = os.path.join('./save_model', 'model')
        saver.save(sess, tensorflow_file, global_step=i, write_meta_graph=False)

        # generate poem
        predict_word = [word2index[utils.start_token]]
        poem = []
        last_state = np.zeros([1, 2, 100], dtype=np.float32)
        for i in range(24):
            state, predict_word = sess.run([model.state, model.predict_word], 
                       feed_dict={model.last_word: predict_word, model.last_state: last_state})
            c, h = state[0], state[1]
            last_state = np.stack([c, h], axis=1)
            word = data.index2word[predict_word[0]]
            if word == utils.end_token:
                break
            poem.append(word)
        print(''.join(poem))
                

