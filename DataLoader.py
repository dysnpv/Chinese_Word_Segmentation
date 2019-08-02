from torch.utils.data import Dataset
from ReadData import read_from_training_data
import torch
import pandas as pd
from Embedding import Embedding
import random
import numpy as np

"""
class SentencesDataset(Dataset):
    def __init__(self, x_tuple, y_list):
        self.x_tuple = x_tuple
        self.y_list = y_list
    def __len__(self):
        return len(self.x_tuple)
    def __getitem__(self, index):
        return self.x_tuple[index], self.y_list[index]
"""        
        
"""
class charactersDataset(Dataset):
    def __init__(self, x_tuple, y_list):
        self.x_tuple = x_tuple
        self.y_list = y_list
    def __len__(self):
        return len(self.x_tuple)
    def __getitem__(self, index):
        return self.x_tuple[index], self.y_list[index]
"""

def create_csv_file(filename, csvfile, num_sentences):
    x_tuple, y_list = read_from_training_data(filename)
    z = list(zip(x_tuple, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)
    with open(csvfile, 'w') as outhandle:
        for i, sentence in enumerate(x_tuple):
            if len(sentence) == 1:
                num_sentences += 1
                continue
            if i % 20 == 0:
                print("Embedded sentences %d / %d." % (i, num_sentences))
            if i >= num_sentences:
                break
            output = Embedding(x_tuple[i])
            X = torch.cat(output[2], 0).transpose(0, 1)
            X = torch.reshape(X, (-1, 768))
            for j in range(output[0].shape[1]):
                assert(torch.equal(X[j * 13 + 12], output[0][0][j]))
            y = np.repeat(np.array(y_tuple[i], dtype = float), 13)
#            print(y)
            y = torch.FloatTensor(y)
            y = torch.reshape(y, (-1, 1))
            IntergratedTensor = torch.cat((X, y), 1)
            assert(IntergratedTensor.shape[1] == 769)
            df = pd.DataFrame(IntergratedTensor.detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))
            
def load_from_csv(csv_file, skip_rows_function, data_processer):
    print('loading from CSV file.')
    IntergratedTensor = torch.tensor(pd.read_csv(csv_file, skiprows = skip_rows_function, header=None).values).float()
#    print(IntergratedTensor.shape)
#    X, y = torch.split(IntergratedTensor, [768, 1], 1)
#    print(X.shape)
#    print(y.shape)
    IntergratedTensor = data_processer(IntergratedTensor)
    print('# of characters: {}'.format(IntergratedTensor.shape[0]))
    return IntergratedTensor

def batch_loader(IntergratedTensor, batch_size):
    def shuffle_rows(T):
        return T[torch.randperm(T.shape[0])]
    shuffled_IntergratedTensor = shuffle_rows(IntergratedTensor)
    X, y = torch.split(shuffled_IntergratedTensor, [shuffled_IntergratedTensor.shape[1] - 1, 1], 1)
#    X = torch.unsqueeze(X, 1)
    y = torch.squeeze(y, 1)
#    print(X.shape)
#    print(y.shape)
    for i in range(0, len(shuffled_IntergratedTensor), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]