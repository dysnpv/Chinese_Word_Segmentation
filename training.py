import torch
import pandas as pd
#from ReadData import read_from_training_data
#from Embedding import Embedding
from os import path
#from sklearn.linear_model import LogisticRegression
#import numpy as np
from DataLoader import create_csv_file, load_from_csv, batch_loader


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
class DropoutClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 200):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout)
        nextout = nextout.clamp(min=0)
        #nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear3(nextout)
        return nextout
    
    def skip_rows(self, i):
        if i % 14 == 12:
            return False
        else:
            return True
    
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        return t

class Dropout_AverageWithFirstLayer_Classifier(DropoutClassifier):
    def skip_rows(self, i):
        if i % 14 == 12 or i % 14 == 0:
            return False
        else:
            return True
    
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        for i in range(0, X.shape[0], 2):
            mean_x = torch.div(torch.add(X[i], X[i + 1]), 2)
            processed_tensor = torch.unsqueeze(torch.cat((mean_x, y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 769]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)
    
class Dropout_AverageLastFourLayers_Classifier(DropoutClassifier):
    def skip_rows(self, i):
        if i % 14 in range(9, 14):
            return False
        else:
            return True
        
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        for i in range(0, X.shape[0], 4):
            mean_x = (X[i] + X[i + 1] + X[i + 2] + X[i + 3]) / 4
            processed_tensor = torch.unsqueeze(torch.cat((mean_x, y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 769]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)
    
class Dropout_ConcatenateWithFirstLayer_Classifier(Dropout_AverageWithFirstLayer_Classifier):
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        for i in range(0, X.shape[0], 2):
            concatenated_x = torch.cat((X[i], X[i + 1]), 0)
            processed_tensor = torch.unsqueeze(torch.cat((concatenated_x, y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 768 * 2 + 1]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)    
    
class Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_Classifier(DropoutClassifier):
    def skip_rows(self, i):
        if i % 14 == 12 or i % 14 == 0:
            return False
        else:
            return True
    
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        the_endofline_tensor = torch.zeros(768, dtype = torch.float)
        for i in range(0, X.shape[0] - 2, 2):
            if torch.equal(X[i + 2], the_endofline_tensor):
#                print('Working')
                continue
            mean_x1 = torch.div(torch.add(X[i], X[i + 1]), 2)
            mean_x2 = torch.div(torch.add(X[i + 2], X[i + 3]), 2)
            processed_tensor = torch.unsqueeze(torch.cat((mean_x1, mean_x2, y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 768 * 2 + 1]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)
    
class Dropout_AverageWithFirstLayer_WithPairEmbedding_Classifier(DropoutClassifier):
    def skip_rows(self, i):
        if i % 14 == 0 or i % 14 == 12 or i % 14 == 13:
            return False
        else:
            return True
    
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        the_endofline_tensor = torch.zeros(768, dtype = torch.float)
        for i in range(0, X.shape[0] - 3, 3):
            if torch.equal(X[i + 2], the_endofline_tensor):
#                print('Working')
                continue
            mean_x = torch.div(torch.add(X[i], X[i + 1]), 2)
            processed_tensor = torch.unsqueeze(torch.cat((mean_x, X[i + 2], y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 768 * 2 + 1]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)

class Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding_Classifier(DropoutClassifier):
    def skip_rows(self, i):
        if i % 14 == 0 or i % 14 == 12 or i % 14 == 13:
            return False
        else:
            return True
    
    def data_processer(self, t):
#        print(t.shape)
#        print(t)
        X, y = torch.split(t, [t.shape[1] - 1, 1], 1)
#        print(X.shape)
#        print(y.shape)
        tensors_list = []
        the_endofline_tensor = torch.zeros(768, dtype = torch.float)
        for i in range(0, X.shape[0] - 3, 3):
            if torch.equal(X[i + 2], the_endofline_tensor):
#                print('Working')
                continue
            mean_x1 = torch.div(torch.add(X[i], X[i + 1]), 2)
            mean_x2 = torch.div(torch.add(X[i + 3], X[i + 4]), 2)
            processed_tensor = torch.unsqueeze(torch.cat((mean_x1, mean_x2, X[i + 2], y[i]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 768 * 3 + 1]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)
"""
def training_scikit_learn(filename):
    num_epochs = 70
    num_test_epochs = 40
    
    x_list, y_list = read_from_training_data(filename)
    myDataset = SentencesDataset(x_list[:int(len(x_list) / 2)], y_list[:int(len(x_list) / 2)])
    myTestset = SentencesDataset(x_list[int(len(x_list) / 2):], y_list[int(len(x_list) / 2):])
    train_loader = torch.utils.data.DataLoader(dataset = myDataset, batch_size = 1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset = myTestset, batch_size = 1, shuffle = True)
    
    x_train_list = []
    y_train_list = []
    epoch_cnt = 0
    for epoch in range(num_epochs):
        sentence, partitions = next(iter(train_loader))
        word_vectors = Embedding(sentence)
        x_train_list.append(torch.squeeze(word_vectors[0], 0))
        y_train_list += partitions
        epoch_cnt += 1
#        print(word_vectors[0].shape)
        print('Epoch:[%d/%d]' % (epoch_cnt, num_epochs))
    x_train_tensor = torch.cat(x_train_list, 0)
    x_train = x_train_tensor.detach().numpy()
    print(x_train.shape)
    y_train = np.asarray(y_train_list)
    y_train = np.reshape(y_train, (-1, 1))
#    print(y_train.shape)
    
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    
    x_test_list = []
    y_test_list = []
    epoch_cnt = 0
    for epoch in range(num_test_epochs):
        sentence, partitions = next(iter(test_loader))
        word_vectors = Embedding(sentence)
        x_test_list.append(torch.squeeze(word_vectors[0], 0))
        y_test_list += partitions
        epoch_cnt += 1
        print('Epoch:[%d/%d]' % (epoch_cnt, num_test_epochs))
    x_test_tensor = torch.cat(x_test_list, 0)
    x_test = x_test_tensor.detach().numpy()
    
    predictions = logisticRegr.predict(x_test)
    positive_cnt = 0
    
    print(predictions.shape)
    for i in range(predictions.shape[0]):
        # Why is this a torch tensor by default?
        positive_cnt += int(predictions[i] == y_test_list[i])
    print('%d out of %d predictions are correct. Accuracy %.4f%%' % (positive_cnt, predictions.shape[0], positive_cnt * 1.0 / predictions.shape[0]))
"""

def train(csv_filename, classifier, test_csv_filename = "", num_epochs = 100, batch_size = 32, learning_rate = 0.001):
    IntergratedTensor = load_from_csv(csv_filename, classifier.skip_rows, classifier.data_processer)
    
    if test_csv_filename == "":
        partition = int(IntergratedTensor.shape[0] * 3 / 4)
        train_data = IntergratedTensor[: partition]
        test_data = IntergratedTensor[partition: ]
    else:
        train_data = IntergratedTensor
        test_data = load_from_csv(test_csv_filename, classifier.skip_rows, classifier.data_processer)
    
    print("Training data size: %d. Testing data size: %d" % (len(train_data), len(test_data)))
    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate) 
    loss = torch.nn.CrossEntropyLoss()   
    
    for epoch in range(num_epochs):
        train_loader = batch_loader(train_data, batch_size)
        total_loss = 0.0
        batch_cnt = 0
        for x_tensor, y_tensor in train_loader:
            optimizer.zero_grad()
#            print(x_tensor.shape)
#            print(y_tensor.shape)
            z = classifier(x_tensor)
            loss_size = loss(z, y_tensor.long())
            loss_size.backward()
            optimizer.step()
            total_loss += loss_size.data.item()
            batch_cnt += 1
        print ('Epoch: [%d/%d], Average Loss: %.4f' % (epoch+1, num_epochs, total_loss / batch_cnt))
        
        test_loader = batch_loader(test_data, batch_size)
        num_characters = 0
        correct_predictions = 0
        for x_tensor, y_tensor in test_loader:
            z = classifier(x_tensor)
            for (i, output) in enumerate(z):
                if y_tensor[i].long() == output.argmax():
                    correct_predictions += 1
                num_characters += 1
        print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
    
def train_simple_Dropout(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = DropoutClassifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_AverageWithFirstLayer_Classifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)

def train_Dropout_AverageLastFourLayers(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_AverageLastFourLayers_Classifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_ConcatenateWithFirstLayer(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_ConcatenateWithFirstLayer_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_WithPairEmbedding(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_AverageWithFirstLayer_WithPairEmbedding_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding(csv_filename, test_csv_filename = "", num_epochs = 100, batch_size = 32, lr = 0.001):
    model = Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding_Classifier(768 * 3, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)