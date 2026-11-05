import numpy as np

# TokenIndicator and BagOfWords from tutorial

class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Concrete implementations will be discussed later")


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector

class BagOfWords(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())
    
    def __call__(self, x: str):
        token_ids = self.tokenizer.encode(x).ids
        # get counts for each id
        vector = np.bincount(token_ids, minlength=len(self)).astype(float)
        return vector


class TfIdf(Encoder):

    def __init__(self, tokenizer: Tokenizer, corpus: list[str]):
        self.tokenizer = tokenizer
        self.bow = BagOfWords(tokenizer)
        self.token_indicator = TokenIndicator(tokenizer)

        # Document frequency: in how many documents does each token appear
        df = np.zeros(len(self))
        for doc in corpus:
            df += self.token_indicator(doc)

        # Smoothed IDF : no division by 0 and ensure not all tokens 
        # that appear everywhere are exactly 0
        n_docs = len(corpus)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        counts = self.bow(x)
        n_tokens = counts.sum()

        # Term frequency: relative frequency within the document
        tf = counts / n_tokens if n_tokens > 0 else counts

        return tf * self.idf
