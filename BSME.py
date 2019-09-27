import torch
import random
from pytorch_transformers import BertTokenizer, BertModel
from training import DropoutClassifier
from ReadData import is_english, read_from_training_data, read_from_testing_data, ReadEnglish

BSME_map = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
Prediction_map = {0: True, 1: False, 2: False, 3: True}

def convert_BSME_train(y):
    BSME_y = []
    for i in range(len(y)):
        BSME_y.append([])
        B_before = False
        for j in range(len(y[i])):
            if B_before:
                if y[i][j] == True:
                    BSME_y[i].append(3)
                    B_before = False
                else:
                    BSME_y[i].append(2)
            else:
                if y[i][j] == True:
                    BSME_y[i].append(0)
                else:
                    BSME_y[i].append(1)
                    B_before = True
    return BSME_y

    
class BSME_model(torch.nn.Module):
    def __init__(self):
        super(BSME_model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 4).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        pooled_output = outputs[2]
        processed_list = []
        assert(len(input_tokens) == pooled_output[0].shape[1])
        for i in range(len(indexed_tokens) - 1):
            mean_1 = (pooled_output[0][0][i] + pooled_output[12][0][i]) / 2
            mean_2 = (pooled_output[0][0][i + 1] + pooled_output[12][0][i + 1]) / 2
            processed_list.append(torch.unsqueeze(torch.cat((mean_1, mean_2), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss
    
def train_BSME(train_file, num_sentences, num_epochs = 15, learning_rate = 0.005, do_save = False, save_path = 'FineTuneModel.bin', eliminate_one = True, language = 'Chinese'):
    model = BSME_model()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one, language)
    torch.cuda.empty_cache()        
    
def data_loader(x_list, y_list):
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)

    for i in range(len(x_tuple)):
        yield x_tuple[i], y_tuple[i]

def prepare_xy_list(filename, num_sentences, language = 'Chinese', eliminate_one = True):
    if language == "English":
        x_list, y_list = ReadEnglish(filename)
    elif language == "Chinese":
        x_list, y_list = read_from_training_data(filename)
    else:
        print("Wrong Language Type! Only suppory Chinese or English.")
        return
    print("There are %d sentences in this %s file." % (len(x_list), language))
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)
    x_list = list(x_tuple)
    y_list = list(y_tuple)
    
    cnt = 0
    l = len(x_list)
    while cnt < l:
        if len(x_list[cnt]) == 0:
            del x_list[cnt]
            del y_list[cnt]
            l -= 1
            continue
        if eliminate_one and len(x_list[cnt]) == 1:
            del x_list[cnt]
            del y_list[cnt]
            l -= 1
            continue
        cnt += 1
    
    return x_list[:num_sentences], convert_BSME_train(y_list[:num_sentences])

def check_BSME_valid(z):
    B_before = False
    for i in range(len(z)):
        if B_before:
            if z[i] == 0 or z[i] == 1:
                return False
            if z[i] == 3:
                B_before = False
        else:
            if z[i] == 2 or z[i] == 3:
                return False
            if z[i] == 1:
                B_before = True
    return True

def train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one, language = 'Chinese'):
    x_list, y_list = prepare_xy_list(train_file, num_sentences, language, eliminate_one)
    
    partition = int(len(x_list) * 4 / 5)

    best_model = model
    best_acc = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loader = data_loader(x_list[:partition], y_list[:partition])
        test_loader = data_loader(x_list[partition:], y_list[partition:])
        batch_cnt = 0
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            batch_cnt += 1
        print ('Epoch: [%d/%d], Average Loss: %.4f' % (epoch+1, num_epochs, total_loss / batch_cnt))
            
        num_characters = 0
        correct_predictions = 0
        correct_segmentation_predictions = 0
        valid_BSME = 0
        total_BSME = 0
        correct_BSME_predictions = 0
        total_BSME_predictions = 0
        for x, y in test_loader:
            z, _ = model(x)
            BSME_predictions = []
            for (i, output) in enumerate(z):
                BSME_predictions.append(output.argmax())
                if y[i] == output.argmax():
                    correct_predictions += 1
                if Prediction_map[y[i]] == Prediction_map[output.argmax().item()]:
                    correct_segmentation_predictions += 1
                num_characters += 1
            assert(check_BSME_valid(y))
            if check_BSME_valid(BSME_predictions) == True:
                valid_BSME += 1
                for i in range(len(BSME_predictions)):
                    if Prediction_map[BSME_predictions[i].item()] == Prediction_map[y[i]]:
                        correct_BSME_predictions += 1
                    total_BSME_predictions += 1
            total_BSME += 1
        print('BSME Classification Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
        print('Segmentation Prediction Accuracy: %.4f' % (correct_segmentation_predictions * 1.0 / num_characters))
        if total_BSME > 0:
            print('Valid BSME Percentage: %.4f' % (valid_BSME * 1.0 / total_BSME))
        if total_BSME_predictions > 0:
            print('Segmentation Prediction Accuracy of Valid BSME: %.4f' % (correct_BSME_predictions * 1.0 / total_BSME_predictions))

        print('\n')
        if correct_predictions * 1.0 / num_characters > best_acc:
            best_acc = correct_predictions * 1.0 / num_characters
            best_model = model
        torch.cuda.empty_cache()
    if do_save:
        torch.save(best_model, save_path)