import torch
from DataLoader import SentencesDataset
from ReadData import read_from_training_data
from Embedding import Embedding
from sklearn.linear_model import LogisticRegression
import numpy as np

"""
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def training(filename):
    x_list, y_list = read_from_training_data(filename)
    myDataset = SentencesDataset(x_list[:int(len(x_list) / 2)], y_list[:int(len(x_list) / 2)])
    
    num_epochs = int(len(x_list) / 2)
    input_size = 768
    output_size = 1
    learning_rate = 0.01
    batchSize = 1
    
    model = LogisticRegression(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
    
    train_loader = torch.utils.data.DataLoader(dataset = myDataset, batch_size = batchSize, shuffle=True)
    
    for epoch in range(num_epochs):
        for sentence, partitions in train_loader:
            word_vectors = Embedding(sentence)
            for i in range(len(sentence)):
#                training_tensor = torch.unsqueeze(torch.stack([torch.squeeze(word_vectors[0][0][i]), torch.squeeze(word_vectors[1])]), 0)
                training_tensor = torch.unsqueeze(torch.squeeze(word_vectors[0][0][i]), 0)
                print(training_tensor.shape)
                optimizer.zero_grad()
                z = model(training_tensor)
                print(z.shape)
                z = torch.squeeze(z, 0)
                print(z.shape)
                print(partitions[i].shape)
                loss = torch.nn.CrossEntropyLoss()(z, partitions[i])
                loss.backward()
                optimizer.step()
            print ('Epoch: [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.data[0]))
"""
      
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

def training(filename):
    x_list, y_list = read_from_training_data(filename)
    X = Embedding(x_list[:100])
        
        