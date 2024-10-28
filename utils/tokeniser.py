"""
Author: Son Phat Tran
This file contains the training logic for the tokeniser
"""
from typing import List
import warnings

from nltk import word_tokenize

# Temporarily ignore warnings
warnings.filterwarnings('ignore')

# Download NLTK
# import nltk
# nltk.download('punkt_tab')


def split(text: str) -> List[str]:
    """
    Split a sentence into a list of tokens
    :param text: sentence to split
    :return: list of tokens
    """
    # Get the paragraph
    paragraphs = [p for p in text.split('\n') if p]

    # Get the result
    result = []

    for paragraph in paragraphs:
        result += word_tokenize(paragraph)
        result.append('\n')

    return result


class Tokeniser:
    def __init__(self, text: str) -> None:
        """
        Text: the text to encode and decode
        :param text: the text
        """
        # Get the individual tokens from the text
        tokens = list(sorted(list(set(split(text)))))

        # Create token => index mapping
        self.word_to_index = {
            word: index for index, word in enumerate(tokens)
        }

        # Create index => token mapping
        self.index_to_word = {
            index: word for index, word in enumerate(tokens)
        }

    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens
        :param text: text to encode
        :return: tokens
        """
        words = split(text)
        return [self.word_to_index[word] for word in words]

    def decode(self, tokens: List[int]) -> str:
        """
        From a list of tokens, retrieve the text
        :param tokens: tokens to decode
        :return: string
        """
        return " ".join([self.index_to_word[token] for token in tokens])

    def __len__(self):
        return len(self.word_to_index)


# Create encoder
with open("../rawdata/data.txt") as file:
    text = file.read()
ENCODER = Tokeniser(text)


if __name__ == "__main__":
    # Define a simple sentence
    sent = "- Anh sẽ ném trụi hết cây xoài nhà em!"

    # Get the vocab size
    print(f"Vocab size: {len(ENCODER)}")

    # Encode
    encoded = ENCODER.encode(sent)
    print(f"Encoded tokens: {encoded}")

    # Decode
    decoded = ENCODER.decode(encoded)
    print(f"Decoded text: {decoded}")