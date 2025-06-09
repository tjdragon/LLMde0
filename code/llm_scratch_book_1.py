print("\nLLM Scratch Book")

import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
from llm_arch import GPT_CONFIG_124M, DummyGPTModel
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# Testing the normalisation layer
from llm_arch import LayerNorm
layer_norm = LayerNorm(emb_dim=5)
out_ln = layer_norm(torch.randn(2,5)) 
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True)
print("LayerNorm output shape:", out_ln.shape)
print("Mean:", mean)
print("Variance:", var)
