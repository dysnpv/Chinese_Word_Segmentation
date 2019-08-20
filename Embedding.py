import torch
from pytorch_transformers import BertTokenizer, BertModel

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#bert_model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True)
bert_model.eval()
bert_model.to('cuda')
    
def Embedding(sentence):

    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    output = bert_model(tokens_tensor.long())
    assert(torch.equal(output[2][len(output[2]) - 1], output[0]))
    return output