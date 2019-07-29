import torch
from pytorch_transformers import BertTokenizer, BertModel

def Embedding(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_model.eval()
#    bert_model.to('cuda')
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
#        tokens_tensor = tokens_tensor.to('cuda')
        
#    print(tokens_tensor.shape)
        
    output = bert_model(tokens_tensor)
        
#    print(len(output))
#    print(output[0].shape)
#    print(output[1].shape)
#    print(torch.stack([torch.squeeze(output[0][0][1]), torch.squeeze(output[1])]).shape)
        
    return output
