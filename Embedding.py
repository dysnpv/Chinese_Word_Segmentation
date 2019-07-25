import torch
from ReadData import SentencesReader
from pytorch_transformers import BertTokenizer, BertModel

def Embedding(filename, file_type):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    x_list = SentencesReader(filename, file_type)
    sentences_tensors = []
    
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_model.eval()
#    bert_model.to('cuda')
    
    for sentence in x_list:
#        print(len(sentence))
        #split into two sentences? use another model?
        indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
        tokens_tensor = torch.tensor([indexed_tokens])
#        tokens_tensor = tokens_tensor.to('cuda')
        
        output = bert_model(tokens_tensor)
        
        sentences_tensors.append(output)
    return sentences_tensors
