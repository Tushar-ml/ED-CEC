from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def get_bert_embeddings(tokens):
    # Tokenize input sequence and get BERT embeddings
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens
    return embeddings

label_map = {'delete': [1,0,0],
             "change": [0,1,0],
             "keep": [0,0,1]}

# Example Usage
aligned_X = ['[UNK]', 'dir', '##ect', '##or', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', 'di', '##d', '[UNK]', 'not', '[UNK]', 'as', '##k', '[UNK]', 'for', '[UNK]', 'a', '[UNK]', 'ca', '##te', '##chi', '##s', '##m', '[UNK]']
aligned_T = ['[UNK]', '[UNK]', '[UNK]', '[UNK]', 'the', '[UNK]', 're', '##ctor', '[UNK]', 'di', '##d', '[UNK]', 'not', '[UNK]', 'as', '##k', '[UNK]', 'for', '[UNK]', 'a', '[UNK]', 'ca', '##te', '##chi', '##s', '##m', '[UNK]']

labels = ['delete', 'delete', 'delete', 'delete', 'change', 'delete', 'change', 'change', 'delete', 'keep', 'keep', 'delete', 'keep', 'delete', 'keep', 'keep', 'delete', 'keep', 'delete', 'keep', 'delete', 'keep', 'keep', 'keep', 'keep', 'keep', 'delete']
labels = torch.tensor([label_map[i] for i in labels])

# Get BERT embeddings for aligned sequences
embeddings_X = get_bert_embeddings(aligned_X)
embeddings_T = get_bert_embeddings(aligned_T)

# Output
print("Embeddings for aligned_X:", embeddings_X)
print("Embeddings for aligned_T:", embeddings_T)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network
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

# Convert embeddings to torch tensors
embeddings_X = torch.tensor(embeddings_X)
embeddings_T = torch.tensor(embeddings_T)

# print(embeddings_X.shape)

# # Create labels for training (example labels, replace with your actual labels)
# labels = torch.tensor([[1, 0, 0],  # "Keep"
#                        [0, 1, 0],  # "Delete"
#                        [0, 0, 1],
#                        [0, 0, 1]]) # "Change"

# Define hyperparameters
input_size = embeddings_X.size(1)  # Size of BERT embeddings
hidden_size = 64
output_size = 3  # Keep, Delete, Change

# Create the model, loss function, and optimizer
model = ClassificationModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(embeddings_X)
    
    # Compute loss
    loss = criterion(outputs, labels.float())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data (embeddings_T)
with torch.no_grad():
    model.eval()
    predictions = model(embeddings_T)

# Convert predictions to labels
predicted_labels = torch.argmax(predictions, dim=1)

# Output
print("Predicted Labels:", predicted_labels.numpy())
