# Simplified Attention Mechanism
print("\nSimplified Attention Mechanism")

import torch

words = [
    "Les",  
    "sanglots",
    "longs",    
    "des",
    "violons"];

inputs = torch.tensor([
    [0.8823, 0.9150, 0.3829], # Les
    [0.9593, 0.3904, 0.6009], # sanglots
    [0.1332, 0.9346, 0.5936], # longs
    [0.1332, 0.9346, 0.5936], # des
    [0.8694, 0.5677, 0.7411], # violons
])

print(f"inputs: {inputs}")

# Let's focus on the second word "sanglots".
# It is also known as the query.

# To find similarity, we can use the dot product between the query and each input.
# If two vectors are similar, their dot product will be high, and if they are not similar, the dot product will be low.

query = inputs[1]  # "sanglots"

attention_score_2 = torch.empty(inputs.shape[0])
for i , x_i in enumerate(inputs):
    attention_score_2[i] = torch.dot(x_i, query)

print(f"attention_score_2: {attention_score_2}")

# We then need to normalize these scores to make them comparable. 
# Naive normalization can be done by dividing each score by the sum of all scores.
attention_score_2_normalised = attention_score_2 / attention_score_2.sum()
print(f"attention_score_2_normalised: {attention_score_2_normalised}")    

# Another way to normalize is to use the softmax function.
# Softmax is a function that converts a vector of scores into probabilities, ensuring that they sum to 1.
# Softmax uses the exponential function to emphasize larger scores and diminish smaller ones.
attention_score_2_softmax = torch.nn.functional.softmax(attention_score_2, dim=0)
print(f"attention_score_2_softmax: {attention_score_2_softmax}")

# Now, we can compute the context vector by taking the weighted sum of the inputs.
context_vector_2 = torch.zeros(inputs.shape[1])
for i, x_i in enumerate(inputs):
    context_vector_2 += attention_score_2_softmax[i] * x_i  
print(f"context_vector_2: {context_vector_2}")

# Just need to do this for all inputs.
attention_scores = inputs @ inputs.T  # Dot product with all inputs
print(f"attention_scores: {attention_scores}")

attention_score_softmax = torch.nn.functional.softmax(attention_scores, dim=1)
print(f"attention_score_softmax: {attention_score_softmax}")

context_vector = attention_score_softmax @ inputs
print(f"context_vector: {context_vector}")

from self_attention import SelfAttentionV1
sa = SelfAttentionV1(input_embedding_dimension=3, output_matrices_dimension=2)
print(f"SelfAttentionV1: {sa(inputs)}")

from self_attention import SelfAttentionV2
sa = SelfAttentionV2(input_embedding_dimension=3, output_matrices_dimension=2)
print(f"SelfAttentionV2: {sa(inputs)}")

from self_attention import CausalSelfAttention
sa = CausalSelfAttention(input_embedding_dimension=3, output_matrices_dimension=2)
print(f"CausalSelfAttention: {sa(inputs)}")

from self_attention import CausalSelfAttentionWithDropouts
batch = torch.stack([inputs, inputs], dim=0)
sa = CausalSelfAttentionWithDropouts(d_in=3, d_out=2, context_length=batch.shape[1], droput=0.0)
print(f"CausalSelfAttentionWithDropouts: {sa(batch)}")

from self_attention import MultiheadAttentionWrapper
batch = torch.stack([inputs, inputs], dim=0)
print(f"Inputs Shape: {inputs.shape}")
print(f"Batch Shape: {batch.shape}")
context_length=batch.shape[1] # nb of tokens in the batch
print(f"context_length: {context_length}")
sa = MultiheadAttentionWrapper(d_in=3, d_out=2, context_length=context_length, num_heads=2, droput=0.0)
print(f"MultiheadAttentionWrapper: {sa(batch)}")


torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

batch_size, context_length, d_in = batch.shape
d_out = 6
from self_attention import MultiHeadAttention
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(f"MultiHeadAttention: {context_vecs}")
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)