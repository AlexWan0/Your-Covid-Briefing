import json
import gensim
from gensim import corpora, models
import numpy as np
import pickle
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from unidecode import unidecode
import re
from tqdm.notebook import tqdm

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")

with open('dictionary.pkl', 'rb') as file_in:
	dictionary = pickle.load(file_in)

lda = gensim.models.LdaMulticore.load('lda.model')

def preprocess(raw):
	result = unidecode(raw)
	result = re.sub(r'(https?:\/\/)?([a-zA-Z0-9-_]{1,}\.){1,}[a-zA-Z0-9-_]{1,}(\/[A-Za-z0-9-._~:?#\[\]@!$&\'()*+,;%=]{1,}){0,}\/?', '', result)
	result = re.sub(r"[^a-zA-Z]", " ", result)
	result = re.sub(' +', ' ', result)
	result = result.lower()
	result = stemmer.stem(result)
	result = [lemmatiser.lemmatize(w) for w in result.split(' ')]
	result = [w for w in result if w not in cachedStopWords]
	result = [w for w in result if len(w) > 4]
	return result

def add_tags(articles):
	for art in articles:
		preprocessed_article = preprocess(art['article_text'])

		bow_article = dictionary.doc2bow(preprocessed_article)
		tags = sorted(lda[bow_article], key=lambda x:-x[1])

		tags = [[a, float(b)] for a, b in tags]

		art['tags'] = tags
		art['max_tag'] = tags[0][0]
		
def add_similarities(articles):
	for art1 in articles:
		similarities = []
		for i, art2 in enumerate(articles):
			if not art1 == art2:
				sim = gensim.matutils.cossim(art1['tags'], art2['tags'])
				similarities.append([i, sim])
		similarities = sorted(similarities, key=lambda x:-x[1])
		art1['sims'] = similarities[:50]
	
if __name__ == '__main__':
	with open('data.json', 'r') as data_in:
		articles = json.load(data_in)
	
	add_tags(articles)
	add_similarities(articles)