import os
import json
from tqdm import tqdm
from entity import FastTextConfig
from entity import Tokenizer

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(file_path, tokenizer: Tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[1]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def load_dataset(path, tokenizer, n_gram_vocab, vocab, pad_size=32):
    buckets = n_gram_vocab
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            label, content = lin.split('\t')
            words = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words.append(vocab.get(word, vocab.get(UNK)))
            bigram = []
            trigram = []
            # ------ngram------
            for i in range(pad_size):
                bigram.append(biGramHash(words, i, buckets))
                trigram.append(triGramHash(words, i, buckets))
            # -----------------
            contents.append((int(label), seq_len, words, bigram, trigram))
    return contents  # [([...], 0), ([...], 1), ...]


def build_dataset(config: FastTextConfig):
    if os.path.exists(config.vocab_path):
        vocab = json.load(open(config.vocab_path, 'r', encoding='utf-8'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=config.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        json.dump(vocab, open(config.vocab_path, 'w', encoding='utf-8'),ensure_ascii=False,indent=4)
    print(f"Vocab size: {len(vocab)}")
    train = load_dataset(config.train_path, config.tokenizer, config.n_gram_vocab, vocab, config.pad_size)
    dev = load_dataset(config.dev_path, config.tokenizer, config.n_gram_vocab, vocab, config.pad_size)
    return vocab, train, dev
