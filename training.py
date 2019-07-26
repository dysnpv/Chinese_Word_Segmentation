import torch
from DataLoader import SentencesDataset
from ReadData import read_from_training_data
from Embedding import Embedding

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