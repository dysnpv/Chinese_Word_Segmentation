from ReadData import read_from_training_data, read_from_testing_data, ReadEnglish
import torch
import pandas as pd
from Embedding import Embedding
import random
import numpy as np

def create_csv_file(filename, csvfile, num_sentences, language_type = "Chinese", is_random = True, eliminate_ones = True):
    if language_type == "Chinese":
        x_tuple, y_list = read_from_training_data(filename)
    elif language_type == "English":
        x_tuple, y_list = ReadEnglish(filename)
    else:
        print("Wrong lauguage type given")
        return
    print("There are %d sentences in this file" % len(x_tuple))
    if is_random:
        z = list(zip(x_tuple, y_list))
        random.shuffle(z)
        x_tuple, y_tuple = zip(*z)
    with open(csvfile, 'w') as outhandle:
        for i, sentence in enumerate(x_tuple):
            if len(sentence) == 0 or (len(sentence) == 1 and eliminate_ones):
                num_sentences += 1
                continue
            if i % 20 == 0:
                print("Embedded sentences %d / %d." % (i, num_sentences))
            if i >= num_sentences:
                break
            
#            print(sentence)
            output = Embedding(x_tuple[i])
            output = list(output)
            output[2] = list(output[2])
            
            pair_characters_layer = []
            for j in range(len(sentence) - 1):
                this_pair = [sentence[j], sentence[j + 1]]
                paired_output = Embedding(this_pair)
                pair_characters_layer.append(torch.reshape(paired_output[1], (1, 1, 768)))
            pair_characters_layer.append(torch.zeros((1, 1, 768), dtype = torch.float).to('cuda'))
            output[2].append(torch.cat(pair_characters_layer, 1))
            X = torch.cat(output[2], 0).transpose(0, 1)
            X = torch.reshape(X, (-1, 768))
            X.to('cuda')
            for j in range(output[0].shape[1]):
                assert(torch.equal(X[j * 14 + 12], output[0][0][j]))
            y = np.repeat(np.array(y_tuple[i], dtype = float), 14)
#            print(y)
            y = torch.FloatTensor(y).to('cuda')
            y = torch.reshape(y, (-1, 1))
            IntergratedTensor = torch.cat((X, y), 1)
            assert(IntergratedTensor.shape[1] == 769)
            df = pd.DataFrame(IntergratedTensor.cpu().detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))
    torch.cuda.empty_cache()
            
def create_test_csv(filename, csv_file, num_sentences):
    x_tuple = read_from_testing_data(filename)
    print("There are %d sentences in this file" % len(x_tuple))
    with open(csv_file, 'w') as outhandle:
        for i, sentence in enumerate(x_tuple):
            if i % 20 == 0:
                print("Embedded sentences %d / %d." % (i, num_sentences))
            if i >= num_sentences:
                break
#            print(sentence)
            output = Embedding(x_tuple[i])
            output = list(output)
            output[2] = list(output[2])
            pair_characters_layer = []
            for j in range(len(sentence) - 1):
                this_pair = [sentence[j], sentence[j + 1]]
                paired_output = Embedding(this_pair)
                pair_characters_layer.append(torch.reshape(paired_output[1], (1, 1, 768)))
            pair_characters_layer.append(torch.zeros((1, 1, 768), dtype = torch.float).to('cuda'))
            output[2].append(torch.cat(pair_characters_layer, 1))
            X = torch.cat(output[2], 0).transpose(0, 1)
            X = torch.reshape(X, (-1, 768))
            for j in range(output[0].shape[1]):
                assert(torch.equal(X[j * 14 + 12], output[0][0][j]))
            df = pd.DataFrame(X.cpu().detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))
    torch.cuda.empty_cache()
            
def load_from_csv(csv_file, skip_rows_function, data_processer):
    print('loading from CSV file.')
    IntergratedTensor = torch.tensor(pd.read_csv(csv_file, skiprows = skip_rows_function, header=None).values).float()
#    print(IntergratedTensor.shape)
#    X, y = torch.split(IntergratedTensor, [768, 1], 1)
#    print(X.shape)
#    print(y.shape)
    IntergratedTensor = data_processer(IntergratedTensor)
    print('# of tokens: {}'.format(IntergratedTensor.shape[0]))
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