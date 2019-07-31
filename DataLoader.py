from torch.utils.data import Dataset
from ReadData import read_from_training_data
import torch
import pandas as pd
from Embedding import Embedding

class SentencesDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
    def __len__(self):
        return len(self.x_list)
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]
    
"""
class charactersDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
    def __len__(self):
        return len(self.x_list)
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]
"""

def create_csv_file(filename, csvfile):
    x_list, y_list = read_from_training_data(filename)
    num_sentences = 10
    with open(csvfile, 'w') as outhandle:
        for i, sentence in enumerate(x_list):
            if i >= num_sentences:
                break
            X = Embedding(x_list[i])[0]
            X = torch.squeeze(X, 0)
            y = torch.FloatTensor(y_list[i])
            y = torch.reshape(y, (-1, 1))
            IntergratedTensor = torch.cat((X, y), 1)
            assert(IntergratedTensor.shape[1] == 769)
            df = pd.DataFrame(IntergratedTensor.detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))