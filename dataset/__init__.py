"""
Author: Son Phat Tran
This file contains the logic for creating the Dataset for training
"""
import torch
from torch.utils.data import Dataset, DataLoader

from utils.tokeniser import ENCODER
from config import CHUNK_SIZE


class TextDataset(Dataset):
    def __init__(self, file_name: str, seq_length: int = CHUNK_SIZE) -> None:
        """
        Initialise the text data loader
        :param file_name: the name of the file
        :param seq_length: size of each chunk (T)
        """
        # Read in the text
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()

        # Encode to tokens
        tokens = ENCODER.encode(text)

        # Spilt into input sequence and output sequence
        x = []
        y = []
        for i in range(len(tokens) - seq_length):
            x.append(tokens[i:i + seq_length])
            y.append(tokens[i+1:i + seq_length + 1])

        # Convert to tensor
        self.x = torch.tensor(x, dtype=torch.long, device='cpu')
        self.y = torch.tensor(y, dtype=torch.long, device='cpu')

    def __len__(self):
        """
        Get the length of the dataset
        :return: the length of the dataset
        """
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    # Create dataset
    text_dataset = TextDataset("../rawdata/data.txt")

    # Create dataloader
    text_dataloader = DataLoader(text_dataset, batch_size=64)
    for inputs, labels in text_dataloader:
        print(f"Input batch shape: {inputs.shape} - Label batch shape: {labels.shape}")

