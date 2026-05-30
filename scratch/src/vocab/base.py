

class BaseTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    @staticmethod
    def train_vocab(vocab_size, file_path=""):
        pass

    @staticmethod
    def load_vocab(vocab_dir):
        pass

