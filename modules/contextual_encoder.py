from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

model_name = 'bert-base-uncased'

word_list = ["battleax", 'rector']


tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def get_bert_embeddings(tokens):
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    print(inputs)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state  # Average pooling over tokens
    return embeddings

embedList = []
for word in word_list:
    embeddings = get_bert_embeddings(tokenizer.tokenize(word))
    embedList.append(embeddings)



embed = torch.vstack(embedList)
print(embed, embed.shape)