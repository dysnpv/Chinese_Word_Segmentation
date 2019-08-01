from torch.utils.data import Dataset
from ReadData import read_from_training_data
import torch
import pandas as pd
from Embedding import Embedding
import random

"""
class SentencesDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
    def __len__(self):
        return len(self.x_list)
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index]
"""        
        
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

def create_csv_file(filename, csvfile, num_sentences):
    x_list, y_list = read_from_training_data(filename)
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_list, y_list = zip(*z)
    
    with open(csvfile, 'w') as outhandle:
        for i, sentence in enumerate(x_list):
            if i % 20 == 0:
                print("Embedded sentences %d / %d." % (i, num_sentences))
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
            
def load_from_csv(csv_file):
    print('loading from CSV file.')
    IntergratedTensor = torch.tensor(pd.read_csv(csv_file).values).float()
#    print(IntergratedTensor.shape)
#    X, y = torch.split(IntergratedTensor, [768, 1], 1)
#    print(X.shape)
#    print(y.shape)
    print('# of characters: {}'.format(IntergratedTensor.shape[0]))
    return IntergratedTensor

def batch_loader(IntergratedTensor, batch_size):
    def shuffle_rows(T):
        return T[torch.randperm(T.shape[0])]
    shuffled_IntergratedTensor = shuffle_rows(IntergratedTensor)
    X, y = torch.split(shuffled_IntergratedTensor, [768, 1], 1)
#    X = torch.unsqueeze(X, 1)
    y = torch.squeeze(y, 1)
#    print(X.shape)
#    print(y.shape)
    for i in range(0, len(shuffled_IntergratedTensor), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]