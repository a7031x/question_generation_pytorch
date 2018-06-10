import json
import os
import utils
import config
import re
from collections import defaultdict

stop_words = set(utils.read_all_lines(config.stopwords_file))

def create_vocab(filename):
    word_vocab = defaultdict(lambda: 0)
    char_vocab = defaultdict(lambda: 0)
    for line in utils.read_all_lines(filename):
        for word in line.split(' '):
            word_vocab[word] += 1
            for char in word:
                char_vocab[char] += 1
    word_vocab = sorted(word_vocab.items(), key=lambda x:-x[1])
    char_vocab = sorted(char_vocab.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.word_vocab_file, ['{}:{}'.format(w,n) for w,n in word_vocab])
    utils.write_all_lines(config.char_vocab_file, ['{}:{}'.format(w,n) for w,n in char_vocab])


def prepare_dataset_with_document(source, target):
    lines = []
    for line in utils.read_all_lines(source):
        sample = json.loads(line)
        documents = sample['documents']
        questions = [sample['segmented_question']] + [doc['segmented_title'] for doc in documents]
        question_words = set(questions[0]) - stop_words
        questions = [' '.join(question) for question in questions]
        for doc in documents:
            for passage in doc['segmented_paragraphs']:
                passage_words = set(passage) - stop_words
                common = question_words & passage_words
                passage = rip_marks(' '.join(passage))
                if len(common) / len(question_words) > 0.3 and len(passage) > 2 * len(questions[0]):
                    lines.append(passage)
                    lines += list(set(questions))
                    lines.append('<P>')
    utils.write_all_lines(target, lines)


def rip_marks(text):
    r = re.sub(r'< ([A-Za-z0-9 /\"=]+) >', r'', text)
    r = re.sub(r'& [a-zA-Z]+ ;', r'', r)
    r = r.replace('　', ' ')
    r = r.replace('\t', ' ')
    r = re.sub(r'  +', r' ', r)
    r = r.strip()
    return r


if __name__ == '__main__':
    prepare_dataset_with_document(config.raw_train_file, config.train_file)
    prepare_dataset_with_document(config.raw_dev_file, config.dev_file)
    create_vocab(config.train_file)
