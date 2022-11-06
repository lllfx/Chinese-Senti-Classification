class Tokenizer(object):

    def __init__(self):
        pass

    def do_call(self, text):
        pass

    def __call__(self, text):
        return self.do_call(text)


class CharTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def do_call(self, text):
        return [c for c in text]


class Config(object):

    def __init__(self):
        self.vocab_path = ''
        self.train_path = ''
        self.dev_path = ''
        self.tokenizer = CharTokenizer()
        self.pad_size = 32
        self.num_classes=2


class FastTextConfig(Config):

    def __init__(self):
        super().__init__()
        self.n_gram_vocab = 2000
