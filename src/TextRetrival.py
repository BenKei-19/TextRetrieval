from datasets import load_dataset
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.spatial.distance import cosine

# Initialize tools
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Load dataset
dataset = load_dataset('ms_marco', 'v1.1')
subset = dataset['test']

# Text preprocessing functions
def tokenize(text):
    return text.split()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]  # Remove stopwords and stem
    return ' '.join(words)

# Create a unique word dictionary
def build_word_dict(corpus):
    word_set = set()
    for doc in corpus:
        normalized_doc = normalize_text(doc)
        word_set.update(tokenize(normalized_doc))
    return list(word_set)

# Vectorize text based on word dictionary
def vectorize_text(text, word_dict):
    normalized_text = normalize_text(text)
    words = tokenize(normalized_text)
    word_count = {word: 0 for word in word_dict}
    for word in words:
        if word in word_count:
            word_count[word] += 1
    return list(word_count.values())

# Build an inverted index with vectors for the corpus
def build_inverted_index(corpus, word_dict):
    inverted_index = {}
    for idx, doc in enumerate(corpus):
        vector = vectorize_text(doc, word_dict)
        inverted_index[(doc, idx)] = vector
    return inverted_index

# Calculate cosine similarity
def calculate_similarity(query_vector, document_vector):
    return 1 - cosine(query_vector, document_vector)

# Rank documents based on query similarity
def rank_documents(query, word_dict, inverted_index):
    query_vector = vectorize_text(query, word_dict)
    scores = []
    for (doc, idx), doc_vector in inverted_index.items():
        similarity_score = calculate_similarity(query_vector, doc_vector)
        scores.append((similarity_score, doc, idx))
    return sorted(scores, reverse=True, key=lambda x: x[0])

# Extract the corpus from the dataset
corpus = []
for sample in subset:
    if sample['query_type'] == 'entity':
        passages = sample['passages']
        corpus.extend(passages['passage_text'])

# Prepare word dictionary and inverted index
word_dict = build_word_dict(corpus)
inverted_index = build_inverted_index(corpus, word_dict)

# Example queries
queries = ['what is the closest planet to the earth']
top_k = 10

# Rank and display results
for query in queries:
    ranked_results = rank_documents(query, word_dict, inverted_index)
    print(f'QUERY: {query}?\n')
    for rank, (score, content, doc_idx) in enumerate(ranked_results[:top_k], start=1):
        print(f'Rank {rank}; Similarity point: {score:.4f}\n{content}\n')