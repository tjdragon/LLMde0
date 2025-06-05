import gensim.downloader as api

model = api.load("word2vec-google-news-300")

word_vectors = model

print(f"Vector for 'computer': {word_vectors['computer']}") 

print(f"Similaraties between man and woman': {word_vectors.similarity('man', 'woman')}")