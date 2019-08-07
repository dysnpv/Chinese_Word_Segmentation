import torch
from pytorch_transformers import BertTokenizer, BertModel

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#bert_model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True)
bert_model.eval()
#    bert_model.to('cuda')
    
def Embedding(sentence):

    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
#    tokens_tensor = tokens_tensor.to('cuda')
        
#    print(tokens_tensor.shape)
        
    output = bert_model(tokens_tensor.long())
        
    """
    print(len(output))
    print(output[0].shape)
    print(output[1].shape)
    print(len(output[2]))
    for i in range(len(output[2])):
        print(output[2][i].shape)
    """
    assert(torch.equal(output[2][len(output[2]) - 1], output[0]))
#    print(torch.stack([torch.squeeze(output[0][0][1]), torch.squeeze(output[1])]).shape)
    """
    test = torch.cat(output[2], 0).transpose(0, 1)
    print(test.shape)    
    test = torch.reshape(test, (-1, 768))
    print(test.shape)
    for i in range(output[0].shape[1]):
        assert(torch.equal(test[i * 13 + 12], output[0][0][i]))
    """
    return output