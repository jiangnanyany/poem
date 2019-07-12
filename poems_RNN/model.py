import tensorflow as tf 

class RNN_poem(object):
    def __init__(self, vocab_size, embedding_size, num_units, test=False):
        self.input_text = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.target_text = tf.placeholder(dtype=tf.int32, shape=[None, None])
        embedding = tf.get_variable('embedding', shape=[vocab_size, embedding_size], dtype=tf.float32)
        input_embed = tf.nn.embedding_lookup(embedding, self.input_text)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, dtype=tf.float32)
        
        outputs, _ = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)
        logits = tf.layers.dense(outputs, vocab_size, name='logit')

        mask = tf.cast(tf.not_equal(self.input_text, 0), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target_text)*mask
        self.loss = tf.reduce_mean(loss)
        
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)
        #if test is True:
        self.last_word = tf.placeholder(dtype=tf.int32, shape=[None])
        self.last_state = tf.placeholder(dtype=tf.float32, shape=[None, 2, num_units])
        last_word_embed = tf.nn.embedding_lookup(embedding, self.last_word)
        output, self.state = cell(last_word_embed, tf.unstack(self.last_state, axis=1))
        logit = tf.layers.dense(output, vocab_size, name='logit', reuse=True)
        self.predict_word = tf.squeeze(tf.multinomial(logit, 1), axis=1)