import re
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocesed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocesed = [item.strip() for item in preprocesed if item.strip()]
        # next is to take care of the end-of-text and missing token markers
        preprocesed = [s if s in self.str_to_int else "<|MIS|>" for s in preprocesed]
        ids = [self.str_to_int[s] for s in preprocesed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text