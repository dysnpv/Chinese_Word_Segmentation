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
#        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#        self.model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        pooled_output = outputs[2]
        processed_list = []
#        print(input_tokens)
#        print(len(input_tokens))
#        print(len(outputs))
#        print(len(pooled_output))
#        print(pooled_output[0].shape)
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

def data_loader(x_list, y_list):
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)

    for i in range(len(x_tuple)):
        yield x_tuple[i], y_tuple[i]

def train(train_file, num_sentences, num_epochs = 60, learning_rate = 0.005, do_save = True, save_path = 'FuneTuneModel.bin', eliminate_one = True):
    x_list, y_list = read_from_training_data(train_file)
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
    
    partition = int(min(len(x_list), num_sentences) * 4 / 5)

    
    model = BertForWordSegmentation()
    best_model = model
    best_acc = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
    for epoch in range(num_epochs):
        train_loader = data_loader(x_list[:partition], y_list[:partition])
        test_loader = data_loader(x_list[partition:min(len(x_list), num_sentences)], y_list[partition:min(len(x_list), num_sentences)])
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
        torch.save(best_model, 'best_model.bin')
        torch.save(model, save_path)

def try_Chinese_model_directly_on_English(model_path, English_file):
    model = torch.load(model_path)
    
    x_list, y_list = ReadEnglish(English_file)
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)
    x_list = list(x_tuple)
    y_list = list(y_tuple)
    
    num_characters = 0
    correct_predictions = 0
    test_loader = data_loader(x_list, y_list)
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