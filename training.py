import torch
from ReadData import is_english, read_from_testing_data
#from Embedding import Embedding
#from sklearn.linear_model import LogisticRegression
#import numpy as np
from DataLoader import load_from_csv, batch_loader


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
    
    def test_data_processor(self, t):
        X = t
        tensors_list = []
        the_endofline_tensor = torch.zeros(768, dtype = torch.float)     
        for i in range(0, X.shape[0] - 3, 3):
            if torch.equal(X[i + 2], the_endofline_tensor):
#                print('Working')
                continue
            mean_x1 = torch.div(torch.add(X[i], X[i + 1]), 2)
            mean_x2 = torch.div(torch.add(X[i + 3], X[i + 4]), 2)
            processed_tensor = torch.unsqueeze(torch.cat((mean_x1, mean_x2, X[i + 2]), 0), 0)
#            print(processed_tensor.shape)
            assert(processed_tensor.shape == torch.Size([1, 768 * 3]))
            tensors_list.append(processed_tensor)
        return torch.cat(tensors_list, 0)

def train(csv_filename, classifier, test_csv_filename = "", num_epochs = 60, batch_size = 96, learning_rate = 0.005, do_test = True):
    IntergratedTensor = load_from_csv(csv_filename, classifier.skip_rows, classifier.data_processer)
    
    if do_test:
        if test_csv_filename == "":
            partition = int(IntergratedTensor.shape[0] * 3 / 4)
            train_data = IntergratedTensor[: partition]
            test_data = IntergratedTensor[partition: ]
        else:
            train_data = IntergratedTensor
            test_data = load_from_csv(test_csv_filename, classifier.skip_rows, classifier.data_processer)
    else:
        train_data = IntergratedTensor
    
    print("Training data size: %d." % len(train_data))
    if do_test:
        print("Testing data size: %d." %  len(test_data))
    
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
        if do_test:        
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
    
def train_simple_Dropout(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = DropoutClassifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageWithFirstLayer_Classifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)

def train_Dropout_AverageLastFourLayers(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageLastFourLayers_Classifier(768, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_ConcatenateWithFirstLayer(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_ConcatenateWithFirstLayer_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_WithPairEmbedding(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageWithFirstLayer_WithPairEmbedding_Classifier(768 * 2, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)
    
def train_Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding(csv_filename, test_csv_filename = "", num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding_Classifier(768 * 3, 2, 200)
    train(csv_filename, model, test_csv_filename, num_epochs, batch_size, lr)

def prepare_script(training_csv, testing_csv, test_file, output_filename = 'result.txt', num_epochs = 60, batch_size = 96, lr = 0.005):
    model = Dropout_AverageWithFirstLayer_ConcatenateWithNextCharacter_WithPairEmbedding_Classifier(768 * 3, 2, 200)
    test_data = load_from_csv(testing_csv, model.skip_rows, model.test_data_processor)
    test_chars = open(test_file).read()
    characters = []
    hash_string = ""
    for c in test_chars:
        if is_english(c):
            hash_string += c
        else:
            if hash_string != "":
                characters.append(hash_string)
                hash_string = ""        
            characters.append(c)
#    print(characters)
    x_list = read_from_testing_data(test_file)
#    print(x_list)
    output = open(output_filename, "w+")
    
    train(training_csv, model, "", num_epochs, batch_size, lr, False)
    
    sentence_id = 0
    token_id = 0
    character_id = 0
    for x_tensor in test_data:
        # if this is the last token:
        while token_id == len(x_list[sentence_id]) - 1:
            output.write(characters[character_id])
#            print("space: %d %c" % (character_id, characters[character_id]))
            output.write("  ")
            if characters[character_id + 1] == '\n':
#                print("end of line: %d %d %d" % (sentence_id, token_id, character_id))
                output.write("\n")
                character_id += 1
            token_id = 0
            sentence_id += 1
            character_id += 1
        z = model(x_tensor)
#        print(z)
#        print(z.argmax())
        output.write(characters[character_id])
        if z.argmax().item() == 1:
            output.write("  ")
        token_id += 1
        character_id += 1
    output.write(characters[character_id])
    output.write("  \n")
    output.close()