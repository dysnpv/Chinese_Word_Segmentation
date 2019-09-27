from pytorch_transformers import BertTokenizer
import nltk.data

def is_english(character):
    if 'A' <= character and character <= 'Z':
        return True
    if 'Ａ' <= character and character <= 'Ｚ':
        return True
    if 'a' <= character and character <= 'z':
        return True
    if 'ａ' <= character and character <= 'ｚ':
        return True
    if '0' <= character and character <= '9':
        return True
    if '０' <= character and character <= '９':
        return True
    if character == '○':
        return True
    if character == '.' or character == '．' or character == '%' or character == '％':
        return True
    return False

def read_from_training_data(filename):
    characters = open(filename).read()
    #x_list is a list of sentences, which means x_list is a 2-d array
    sentence_cnt = 0
    x_list = [[]]
    y_list = [[]]
    i = 0
    english_before = False
    for i in range(len(characters)):
        if is_english(characters[i]):
            if english_before:
                continue
            else:
                english_before = True
                continue
        else:
            if english_before:
                x_list[sentence_cnt].append('#')
                y_list[sentence_cnt].append(characters[i] == ' ')
                english_before = False
                
        if not characters[i] == ' ':
            if characters[i] == '\n':
                if not len(x_list[sentence_cnt]) == 0:
                    sentence_cnt += 1
                    x_list.append([])
                    y_list.append([])
                continue
        
            y_list[sentence_cnt].append((characters[i + 1] == ' '))
            x_list[sentence_cnt].append(characters[i])
            
            if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
                
    i = 0
    l = len(x_list)
    while i < l:
        if(len(x_list[i])) > 512:
            del x_list[i]
            del y_list[i]
            i -= 1
            l -= 1
        i += 1
            
    if len(x_list[len(x_list) - 1]) == 0:
        del x_list[len(x_list) - 1]
        del y_list[len(x_list) - 1]
    return x_list, y_list

def read_from_testing_data(filename):
    characters = open(filename).read()
    sentence_cnt = 0
    x_list = [[]]
    english_before = False
    for i in range(len(characters)):
        if is_english(characters[i]):
            if english_before:
                continue
            else:
                english_before = True
                continue
        else:
            if english_before:
                x_list[sentence_cnt].append('#')
                english_before = False
        if characters[i] == '\n':
            if not len(x_list[sentence_cnt]) == 0:
                sentence_cnt += 1
                x_list.append([])
            continue
        x_list[sentence_cnt].append(characters[i])
        
        if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
            sentence_cnt += 1
            x_list.append([])
        
    return [x for x in x_list if x != []]

def sentenceReader(filename, file_type):
    assert(file_type == 'testing' or file_type == 'training')
    if(file_type == 'testing'):
        return read_from_testing_data(filename)
    if(file_type == 'training'):
        x_list, _ = read_from_training_data(filename)
        return x_list
    
def ReadEnglish(filename):
    characters = open(filename, 'r', encoding = 'utf-8').read()    
    split_tool = nltk.data.load('tokenizers/punkt/english.pickle')
#    split_tool = nltk.data.load('tokenizers/punkt/finnish.pickle')
    sentences = split_tool.tokenize(characters)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
    i = 0
    x_list = []
    y_list = []
    while i < len(sentences):
        if '#' in sentences[i]:
            del sentences[i]
            continue
        x_list.append([])
        y_list.append([])
        tokenized_text = tokenizer.tokenize(sentences[i])
        x_list[i] = tokenized_text
        for j in range(1, len(tokenized_text)):
            if len(tokenized_text) >= 2 and tokenized_text[j][:2] == "##":
                y_list[i].append(False)
            else:
                y_list[i].append(True)
        y_list[i].append(True)
        i += 1
    i = 0
    l = len(x_list)
    while i < l:
        if(len(x_list[i])) > 512:
            del x_list[i]
            del y_list[i]
            l -= 1
            continue
        
        if(len(x_list[i])) == 1:
            del x_list[i]
            del y_list[i]
            l -= 1
            if i < l:
                del x_list[i]
                del y_list[i]
                l -= 1
            if i >= 1:
                del x_list[i - 1]
                del y_list[i - 1]
                i -= 1
                l -= 1
            continue
        i += 1
    return x_list, y_list