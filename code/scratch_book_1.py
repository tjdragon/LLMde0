import re

print("\nScratch Book for Tokenization")

from simple_tokenizer import SimpleTokenizer

with open("./data/TinyShakespeare.txt", "r") as f:
    raw_text = f.read()
    raw_text = raw_text.upper()
    
print("Length of raw text:", len(raw_text))

preprocesed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocesed_text = [item.strip() for item in preprocesed_text if item.strip()]
print("Length of preprocessed text:", len(preprocesed_text))
print(preprocesed_text[:10]) 

all_words = sorted(set(preprocesed_text)) # 12002 tokens
all_words.extend(["<|EOT|>", "<|MIS|>"]) # Adding end-of-text and missing token markers, 12004 tokens
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
print("Vocabulary sample:", list(vocab.items())[:10])

tokenizer = SimpleTokenizer(vocab)
text_2_encode = "You are all resolved rather to die than to famish?".upper()
print("Text to encode:", text_2_encode)
encoded_text = tokenizer.encode(text_2_encode)
print("Encoded text:", encoded_text)
print("Decoded text:", tokenizer.decode(encoded_text))

text2_2_encode = " Hi. Yo. Hello! How are you? I am fine, thank you.".upper()

final_text_2_encode = " <|EOT|> ".join([text_2_encode, text2_2_encode])
encoded_text2 = tokenizer.encode(final_text_2_encode)
print("Encoded text:", encoded_text2)
print("Decoded text:", tokenizer.decode(encoded_text2))

# BPE
print("\nBPE Tokenization")
import importlib.metadata
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))
bpe_tokenizer = tiktoken.get_encoding("gpt2")
bpe_encoded_text = bpe_tokenizer.encode(text_2_encode)
print("BPE encoded text:", bpe_encoded_text)
print("BPE decoded text:", bpe_tokenizer.decode(bpe_encoded_text))

# Input - Target Pairs
print("\nInput - Target Pairs")
raw_text_encoded = bpe_tokenizer.encode(raw_text)
print("Length of raw text encoded:", len(raw_text_encoded))

context_length = 4
x = raw_text_encoded[:context_length]
y = raw_text_encoded[1:context_length + 1]
print("Input (x):", x)
print("Target (y):", y)
for i in range(1, context_length + 1):
    context = raw_text_encoded[:i]
    target = raw_text_encoded[i]
    print(context, "-->", target)
    print(bpe_tokenizer.decode(context), "-->", bpe_tokenizer.decode([target]))

# Data Loaders
print("\nData Loaders")

from data_loader import create_dataloader
data_loader = create_dataloader(raw_text, batch_size=4, context_length=1, stride=4, shuffle=True, drop_last=True, num_workers=0)
data_iter = iter(data_loader)
first_batch = next(data_iter)
print("First batch input IDs:", first_batch)