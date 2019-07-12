import collections
import numpy as np 

start_token = 'B'
end_token = 'E'
PADDING = '**PAD**'

class Dataset(object):
    def __init__(self, filePath):
        poems = []
        with open(filePath, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    title, content = line.strip().split(':')
                    content = content.replace(' ', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 79:
                        continue
                    content = start_token + content + end_token
                    poems.append(content)
                except ValueError as e:
                    pass
        all_words = [word for poem in poems for word in poem]
        counter = collections.Counter(all_words)
        words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
        word2index = dict(zip(words, range(1, 1+len(words))))
        word2index[PADDING] = 0
        self.num_items = len(poems)
        self.word2index = word2index
        self.index2word = dict([(v, k) for k, v in word2index.items()])
        self.poems = [[word2index[word] for word in poem] for poem in poems]

    def get_batch(self, batch_size):
        from_ = 0
        while True:
            to = from_ + batch_size
            poems = self.poems[from_: to]
            max_sentence_length = np.max([len(poem) for poem in poems])
            batch = np.full((len(poems), max_sentence_length), 0, dtype=np.int32)
            for row, poem in enumerate(poems):
                batch[row, :len(poem)] = poem
            x_data = batch[:, :-1]
            y_data = batch[:, 1:]
            yield x_data, y_data
            from_ = to
            if from_ >= self.num_items:
                break