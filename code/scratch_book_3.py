# Token Embeddings
print("\nCreating Token Embeddings")

import torch

# I  love golf a  lot  -- raw text - vocab size is 5 
# 2, 3,   4,   5, 1    -- token ids  
input_ids = torch.tensor([2, 3, 4])  # I  love golf

vocab_size = 5
feature_size = 3 # for example - GPT-2 has 768 features 

torch.manual_seed(42)  # For reproducibility

embedding_layer = torch.nn.Embedding(vocab_size, feature_size)
print(f"Embedding Layer Weights: {embedding_layer.weight}")


# Positional Embeddings
print("\nCreating Positional Embeddings")
vocab_size = 50257
feature_size = 256 
max_length = 4

embedding_layer = torch.nn.Embedding(vocab_size, feature_size)

from data_loader import create_dataloader
with open("./data/TinyShakespeare.txt", "r") as f:
    raw_text = f.read()
    raw_text = raw_text.upper()
data_loader = create_dataloader(raw_text, batch_size=8, context_length=max_length, stride=max_length, shuffle=False)
dara_iter = iter(data_loader)
inputs, targets = next(dara_iter)

print("Token IDs:", inputs)
print("Inputs shape:", inputs.shape)

token_embeddings = embedding_layer(inputs)
print("Token Embeddings shape:", token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, feature_size)
pos_embedding = pos_embedding_layer(torch.arange(max_length))
print("Positional Embeddings shape:", pos_embedding.shape)

input_embeddings = token_embeddings + pos_embedding
print("Input Embeddings shape:", input_embeddings.shape)