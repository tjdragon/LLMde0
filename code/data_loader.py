import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, txt, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))  
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
# Create a DataLoader for the TextDataset
# Parameters: 
# - raw_txt: the raw text to be tokenized
# - batch_size: size of each batch 
# - context_length: maximum length of the input sequence
# - stride: step size for creating input-target pairs
# - shuffle: whether to shuffle the dataset
# - drop_last: whether to drop the last incomplete batch
# - num_workers: number of subprocesses to use for data loading
def create_dataloader(raw_txt, batch_size=4, context_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    from tiktoken import get_encoding
    tokenizer = get_encoding("gpt2")
    
    dataset = TextDataset(raw_txt, tokenizer, context_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader