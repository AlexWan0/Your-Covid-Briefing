import nltk
nltk.download('punkt')

from goose3 import Goose
from dateutil import parser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import time
import numpy as np
import pickle
from datetime import datetime
from datetime import timedelta
import re
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm.notebook import tqdm
from unidecode import unidecode
import requests
from lda_engine import add_tags, add_similarities

g = Goose({'enable_image_fetching':False, 'browser_user_agent': 'Mozilla'}) # init goose object

# init summarizer objects
stemmer = Stemmer('english')
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words('english')

def load_model(model_path):
	interpreter = tf.lite.Interpreter(model_path=model_path)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	return interpreter, input_details, output_details

def load_tokenizer(tokenizer_path):
	with open(tokenizer_path, 'rb') as file_in:
		tokenizer = pickle.load(file_in)
	return tokenizer

# init model
interpreter, input_details, output_details = load_model('model_0.1567.tflite')
tokenizer = load_tokenizer('tokenizer_0.1567.pkl')

# extract text from article url
def extract_text(article_url):
	article = g.extract(url=article_url)
	return article

p_stemmer = PorterStemmer()
wn_lemmatiser = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")
  
def process_text(text):
  # unicode to ascii
  result = unidecode(text)

  # remove anything that's not a letter
  result = re.sub(r"[^a-zA-Z]", " ", result)

  # remove any consecutive spaces
  result = re.sub(' +', ' ', result)

  # lowercase
  result = result.lower()

  # remove stop words
  results = ' '.join([w for w in result.split(' ') if w not in cachedStopWords])

  # stem
  result = p_stemmer.stem(results)

  # lemmatise
  result = ' '.join([wn_lemmatiser.lemmatize(w) for w in result.split(' ')])

  # remove beginning and end spaces
  result = result.strip()

  return result

def run_engine():
	# get news from last 24 hours
	all_articles = []

	for q in ['coronavirus', 'covid-19']:
		url = 'http://newsapi.org/v2/everything?q=%s&pageSize=100&apiKey=' % q
		print(url)
		response = requests.get(url)
		res = response.json()
		if not res['articles'] == None:
			all_articles.extend(res['articles'])

	headline_set = set()
	article_norepeat = []

	for art in all_articles:
		title = art['title']
		if not title in headline_set:
			headline_set.add(title)
			article_norepeat.append(art)

	all_articles = article_norepeat

	print(len(all_articles))

	# get article content/data from urls
	num_skipped = 0
	text_objects = []

	for i, a in enumerate(all_articles):
		ts1 = time.time()
		try:
			article_goose = extract_text(a['url'])
			data = article_goose.infos
			data['newsapi'] = a
			text_objects.append(data)
		except Exception as e:
			print(e)
			num_skipped += 1
		print('%d: %.2f' % (i, time.time() - ts1))

	# convert scraped dicts to smaller dicts used by the server
	server_objects = []

	for to in text_objects:
		server_to = {}
		server_to['title'] = to['newsapi']['title']
		server_to['publisher'] = to['newsapi']['source']['name']
		server_to['date'] = parser.parse(to['newsapi']['publishedAt']).strftime('%m/%d/%y')
		server_to['url'] = to['newsapi']['url']
		server_to['article_text'] = to['cleaned_text']
		if len(server_to['article_text']) > 0 and len(server_to['title']) > 0:
			server_objects.append(server_to)

	# make summaries
	for i, s_to in enumerate(server_objects):
		article_text = s_to['article_text']

		pt_parser = PlaintextParser.from_string(article_text, SumyTokenizer('english'))
		
		summarized = ''
		
		for sentence in summarizer(pt_parser.document, 10):
			summarized += str(sentence) + ' '
		
		server_objects[i]['summary'] = summarized
		server_objects[i]['read_time'] = round(len(summarized.split(' ')) / 200, 1)

	# find neutrality
	for i, s_to in enumerate(server_objects):
		ts1 = time.time()

		article_text = s_to['article_text']

		article_text = process_text(article_text)

		#print(article_text)

		article_text = tokenizer.texts_to_sequences([article_text])

		article_text = sequence.pad_sequences(article_text, maxlen=400)

		#print(article_text)

		interpreter.set_tensor(input_details[0]['index'], np.float32(article_text, axis=0))

		interpreter.invoke()

		output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]

		#print(output_data)

		server_objects[i]['neutrality'] = float(output_data)
		
		print('%d: %.2f' % (i, time.time() - ts1))

	add_tags(server_objects)
	add_similarities(server_objects)

	# dump data
	with open('data.json', 'w') as data_out:
		json.dump(server_objects, data_out)

if __name__ == '__main__':
	run_engine()