import torch

def read_from_training_data(filename):
    characters = open(filename).read()
    #x_list is a list of sentences, which means x_list is a 2-d array
    sentence_cnt = 0
    x_list = [[]]
    y_list = [[]]
    for i in range(len(characters)):
        if not characters[i] == ' ':
            if characters[i] == '\n':
                if not len(x_list[sentence_cnt]) == 0:
                    sentence_cnt += 1
                    x_list.append([])
                    y_list.append([])
                continue
            
            if len(x_list[sentence_cnt]) >= 300:
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
            
            y_list[sentence_cnt].append((characters[i + 1] == ' '))
            x_list[sentence_cnt].append(characters[i])
            if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
    return x_list, y_list

def read_from_testing_data(filename):
    characters = open(filename).read()
    sentence_cnt = 0
    x_list = [[]]
    for i in range(len(characters)):
        if characters[i] == '\n':
            if not len(x_list[sentence_cnt]) == 0:
                sentence_cnt += 1
                x_list.append([])
            continue
        x_list[sentence_cnt].append(characters[i])
        if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
            sentence_cnt += 1
            x_list.append([])
    return x_list

def sentenceReader(filename, file_type):
    assert(file_type == 'testing' or file_type == 'training')
    if(file_type == 'testing'):
        return read_from_testing_data(filename)
    if(file_type == 'training'):
        x_list, _ = read_from_training_data(filename)
        return x_list