import config
import utils
import random
import numpy as np

class Dataset(object):
    def __init__(self):
        self.w2i, self.i2w, self.wi2n = load_vocab(config.word_vocab_file, config.word_vocab_size)
        self.c2i, self.i2c, self.ci2n = load_vocab(config.char_vocab_file, config.char_vocab_size)
        self.words = list(self.w2i.keys())
        self.chars = list(self.c2i.keys())
        self.word_weights = [self.wi2n[id] for id in range(len(self.words))]
        self.norm_word_weights = self.word_weights / np.sum(self.word_weights)
        self.train_set = load_qa(config.train_file, config.answer_limit)
        self.dev_set = load_qa(config.dev_file, config.answer_limit)
        self.stopwords = set(utils.read_all_lines(config.stopwords_file))


class Feeder(object):
    def __init__(self, dataset):
        self.dataset = dataset


    def word_to_id(self, word):
        if word in self.dataset.c2i:
            return self.dataset.c2i[word]
        else:
            return config.OOV_ID


    def sent_to_ids(self, sent):
        return [self.word_to_id(w) for w in sent]



    def ids_to_sent(self, ids):
        return ''.join([self.dataset.i2c[id] for id in ids if id != config.NULL_ID])


    def label_vector(self, sent):
        v = [0] * len(self.dataset.c2i)
        for char in sent:
            if char in self.dataset.c2i:
                v[self.dataset.c2i[char]] = 1
        return v


    def seq_tag(self, question, answer):
        return [1 if char in question else 0 for char in answer]


    def decode_logit(self, logit):
        ids = np.argmax(logit, -1)
        sent = self.ids_to_sent(ids)
        return sent


class TrainFeeder(Feeder):
    def __init__(self, dataset):
        super(TrainFeeder, self).__init__(dataset)


    def prepare(self, type):
        if type == 'train':
            self.prepare_data(self.dataset.train_set)
            self.keep_prob = config.keep_prob
            random.shuffle(self.data)
        elif type == 'dev':
            self.prepare_data(self.dataset.dev_set)
            self.keep_prob = 1.0
        self.cursor = 0
        self.size = len(self.data)


    def prepare_data(self, dataset):
        self.data = dataset
        r = set()
        for _, questions in self.data:
            for question in questions:
                r.add(tuple(question))
        self.questions = [list(q) for q in r]


    def create_record(self, example):
        passage, questions = example
        truth = questions
        false = self.create_false_questions(questions)
        record = passage, truth + false, [1] * len(truth) + [0] * len(false)
        return record


    def create_false_questions(self, questions, multipler=1):
        weight = [1, 0, 1, 4]
        r = []
        for _ in range(len(questions)*multipler):
            id = np.random.choice(len(weight), None, p=weight/np.sum(weight))
            if id == 0:
                q = self.shuffle_question(questions)
            elif id == 1:
                q = self.duplicate_word(questions)
            elif id == 2:
                q = self.insert_word(questions)
            else:
                q = self.sample_other_question(questions)
            if q is not None:
                r.append(q)
        return r


    def shuffle_question(self, questions):
        questions = [q for q in questions if len(q) >= 4]
        if not questions:
            return None
        question = random.sample(questions, 1)[0].copy()
        question = [list(w) for w in question]
        for w in question:
            random.shuffle(w)
        question = [''.join(w) for w in question]
        random.shuffle(question)
        return question


    def duplicate_word(self, questions):
        question = random.sample(questions, 1)[0].copy()
        words = random.sample(question, random.randint(2, 3))
        for word in words:
            question.insert(random.randint(0, len(question)), word)
        return question


    def insert_word(self, questions):
        question = random.sample(questions, 1)[0].copy()
        #words = random.sample(self.dataset.words, random.randint(1, 2))
        words = np.random.choice(self.dataset.words, p=self.dataset.norm_word_weights, size=random.randint(2, 4))
        for word in words:
            question.insert(random.randint(0, len(question)), word)
        return question


    def sample_other_question(self, questions):
        while True:
            question = random.sample(self.questions, 1)[0]
            question_words = self.nonstopwords(question)
            if np.alltrue([len(question_words & self.nonstopwords(q)) == 0 for q in questions]):
                return question


    def nonstopwords(self, sent):
        return set(sent) - self.dataset.stopwords


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size=config.batch_size):
        size = min(self.size - self.cursor, batch_size)
        batch = self.data[self.cursor:self.cursor+size]
        batch_pid = []
        batch_qid = []
        batch_label = []
        for example in batch:
            passage, questions, labels = self.create_record(example)
            passage, questions = ''.join(passage), [''.join(q) for q in questions]
            pids = self.sent_to_ids(passage)
            qids = [self.sent_to_ids(question) for question in questions]
            #question_vector = self.label_vector(questions[0])
            #passage_tag = [self.seq_tag(q, p) for p,q in zip()]
            batch_pid.append(pids)
            batch_qid.append(qids)
            batch_label.append(labels)
        self.cursor += size
        return align2d(batch_pid), align3d(batch_qid), align2d(batch_label), self.keep_prob


def load_vocab(filename, count):
    w2i = {
        config.NULL: config.NULL_ID,
        config.OOV: config.OOV_ID,
        config.SOS: config.SOS_ID,
        config.EOS: config.EOS_ID
    }
    i2c = {
        config.NULL_ID: 0,
        config.SOS_ID: 0,
        config.EOS_ID: 0
    }
    all_entries = list(utils.read_all_lines(filename))
    count -= len(w2i)
    count = min(count, len(all_entries))
    for line in all_entries[:count]:
        word, freq = line.rsplit(':', 1)
        id = len(w2i)
        w2i[word] = id
        i2c[id] = int(freq)
    i2w = {k:v for v,k in w2i.items()}
    i2c[config.OOV_ID] = len(all_entries) - count
    return w2i, i2w, i2c


def load_qa(filename, answer_limit=0):
    lines = []
    r = []
    for line in utils.read_all_lines(filename):
        if line == '<P>':
            passage = lines[0].split(' ')
            if len(''.join(passage)) <= config.max_passage_len:
                questions = [q.split(' ') for q in lines[1:] if len(q.replace(' ', '')) <= config.max_question_len]
                if questions:
                    r.append((passage, questions))
            lines.clear()
        else:
            lines.append(line)
    return r


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [row + [fill] * (mlen - len(row)) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values


if __name__ == '__main__':
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    feeder.prepare('dev')
    pids, qids, labels, keep_prob = feeder.next()
    for pid, qid, label in zip(pids, qids, labels):
        print('--------------------')
        passage = feeder.ids_to_sent(pid)
        questions = [feeder.ids_to_sent(q) for q in qid]
        for t, question in zip(label, questions):
            if question:
                print('{} {}'.format(t, question))
