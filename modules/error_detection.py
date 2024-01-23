from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings
import torch
import torch.nn as nn

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
embeddings = BertEmbeddings(bert_model.config)

def get_bert_embeddings(tokens):
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

def token_positional_embeddings(tokens, return_embed = False):
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)

    if return_embed:
        with torch.no_grad():
            result = embeddings(input_ids = inputs['input_ids'],
                                token_type_ids = inputs['token_type_ids'])
        return result
    
    input_shape = inputs['input_ids'].size()
    position_ids = embeddings.position_ids[:, : input_shape[1]]

    token_type_embeddings = embeddings.token_type_embeddings(inputs['token_type_ids'])
    position_embeddings = embeddings.position_embeddings(position_ids)

    return token_type_embeddings, position_embeddings

class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x