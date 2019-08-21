#from .modeling_utils import PreTrainedModel
import torch
import random
from pytorch_transformers import BertTokenizer, BertModel
from training import DropoutClassifier
from ReadData import is_english, read_from_training_data, read_from_testing_data, ReadEnglish


class BertForWordSegmentation(torch.nn.Module):
    def __init__(self):
        super(BertForWordSegmentation, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
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

class BertForWordSegmentation_4(torch.nn.Module):
    def __init__(self):
        super(BertForWordSegmentation_4, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        
        self.classifier = DropoutClassifier(768 * 4, 2).to('cuda')
    def forward(self, input_tokens, labels = None):
        bert_tokens = []
        bert_tokens.append("[CLS]")
        bert_tokens += input_tokens
        bert_tokens.append("[SEP]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        pooled_output = outputs[2]
        processed_list = []
        assert(len(bert_tokens) == pooled_output[0].shape[1])
        for i in range(len(indexed_tokens) - 3): 
            mean_1 = (pooled_output[0][0][i] + pooled_output[12][0][i]) / 2
            mean_2 = (pooled_output[0][0][i + 1] + pooled_output[12][0][i + 1]) / 2
            mean_3 = (pooled_output[0][0][i + 2] + pooled_output[12][0][i + 2]) / 2
            mean_4 = (pooled_output[0][0][i + 3] + pooled_output[12][0][i + 3]) / 2
            processed_list.append(torch.unsqueeze(torch.cat((mean_1, mean_2, mean_3, mean_4), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss        

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
    
    return x_list[:num_sentences], y_list[:num_sentences]

def train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one):
    x_list, y_list = prepare_xy_list(train_file, num_sentences, 'Chinese', eliminate_one)
    
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
        for x, y in test_loader:
            z, _ = model(x)
            for (i, output) in enumerate(z):
                if y[i] == output.argmax():
                    correct_predictions += 1
                num_characters += 1
        print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
        if correct_predictions * 1.0 / num_characters > best_acc:
            best_acc = correct_predictions * 1.0 / num_characters
            best_model = model
        torch.cuda.empty_cache()
    if do_save:
        torch.save(best_model, save_path)

def test_model(model_path, filename, num_sentences, language = "English"):
    model = torch.load(model_path)
    
    x_list, y_list = prepare_xy_list(filename, num_sentences, language)
    
    num_characters = 0
    correct_predictions = 0
    test_loader = data_loader(x_list[:num_sentences], y_list[:num_sentences])
    for x, y in test_loader:
        z, _ = model(x)
        for (i, output) in enumerate(z):
            if y[i] == output.argmax():
                correct_predictions += 1
            num_characters += 1
    print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
    
def prepare_script(model_path, test_file, output_filename = 'FineTune_result.txt'):
    model = torch.load(model_path)
    
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
            
    x_list = read_from_testing_data(test_file)
    output = open(output_filename, "w+")

    character_id = 0
    for x in x_list:
        if len(x) == 0:
            print("Error! Sentecen with length 0!")
            continue
        if len(x) > 1:
            z, _ = model(x)
    #        print(z)
    #        print(z.argmax())
            for (i, prediction) in enumerate(z):
                output.write(characters[character_id])
                if prediction.argmax().item() == 1:
                    output.write("  ")
                character_id += 1
        output.write(characters[character_id])
        character_id += 1
#       print("space: %d %c" % (character_id, characters[character_id]))
        output.write("  ")
        if characters[character_id] == '\n':
#                print("end of line: %d %d %d" % (sentence_id, token_id, character_id)).
            output.write("\n")
            character_id += 1
    torch.cuda.empty_cache()
    output.close()
        
def train_on_two_languages(train_chinese_file, num_chinese_sentences, train_english_file, num_english_sentences, num_epochs = 15, learning_rate = 0.005, do_save = True, save_path = 'TwoLanguageModel', eliminate_one = True):
    chinese_x, chinese_y = prepare_xy_list(train_chinese_file, num_chinese_sentences, 'Chinese', eliminate_one)
    english_x, english_y = prepare_xy_list(train_english_file, num_english_sentences, 'English', eliminate_one)
        
    partition = int(len(chinese_x) * 4 / 5)
    train_x = chinese_x[:partition]
    train_y = chinese_y[:partition]
    test_chinese_x = chinese_x[partition:]
    test_chinese_y = chinese_y[partition:]
    
    partition = int(len(english_x) * 4 / 5)
    train_x += english_x[:partition]
    train_y += english_y[:partition]
    test_english_x = english_x[partition:]
    test_english_y = english_y[partition:]
    
    
    model = BertForWordSegmentation()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
    
    best_chinese_model = model
    best_chinese_acc = 0.0
    best_english_model = model
    best_english_acc = 0.0
    best_overall_model = model
    best_overall_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loader = data_loader(train_x, train_y)
        test_chinese_loader = data_loader(test_chinese_x, test_chinese_y)
        test_english_loader = data_loader(test_english_x, test_english_y)
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
            
        num_chinese_tokens = 0
        correct_chinese_predictions = 0
        for x, y in test_chinese_loader:
            z, _ = model(x)
            for (i, output) in enumerate(z):
                if y[i] == output.argmax():
                    correct_chinese_predictions += 1
                num_chinese_tokens += 1
        print('%d Chinese tokens' % num_chinese_tokens)
        print('Test Accuracy of Chinese: %.4f' % (correct_chinese_predictions * 1.0 / num_chinese_tokens))
        if correct_chinese_predictions * 1.0 / num_chinese_tokens > best_chinese_acc:
            best_chinese_acc = correct_chinese_predictions * 1.0 / num_chinese_tokens
            best_chinese_model = model
            
        num_english_tokens = 0
        correct_english_predictions = 0
        for x, y in test_english_loader:
            z, _ = model(x)
            for (i, output) in enumerate(z):
                if y[i] == output.argmax():
                    correct_english_predictions += 1
                num_english_tokens += 1
        print('%d English tokens' % num_english_tokens)
        print('Test Accuracy of English: %.4f' % (correct_english_predictions * 1.0 / num_english_tokens))
        if correct_english_predictions * 1.0 / num_english_tokens > best_english_acc:
            best_english_acc = correct_english_predictions * 1.0 / num_english_tokens
            best_english_model = model
        
        print('Test Accuracy of all tokens: %.4f\n' % ((correct_chinese_predictions + correct_english_predictions) * 1.0 / (num_chinese_tokens + num_english_tokens)))
        if (correct_chinese_predictions + correct_english_predictions) * 1.0 / (num_chinese_tokens + num_english_tokens) > best_overall_acc:
            best_overall_acc = (correct_chinese_predictions + correct_english_predictions) * 1.0 / (num_chinese_tokens + num_english_tokens)
            best_overall_model = model
            
        torch.cuda.empty_cache()
    if do_save:
        torch.save(best_chinese_model, save_path + '_Chinese.bin')
        torch.save(best_english_model, save_path + '_English.bin')
        torch.save(best_overall_model, save_path + '_Overall.bin')
        
def train_2(train_file, num_sentences, num_epochs = 15, learning_rate = 0.005, do_save = True, save_path = 'FineTuneModel.bin', eliminate_one = True):
    model = BertForWordSegmentation()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()
    
def train_4(train_file, num_sentences, num_epochs = 15, learning_rate = 0.005, do_save = True, save_path = 'FineTuneModel.bin', eliminate_one = True):
    model = BertForWordSegmentation_4()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()