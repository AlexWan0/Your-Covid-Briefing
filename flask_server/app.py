from flask import Flask, render_template, request, jsonify
import json
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from news_engine import run_engine

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

articles = []

articles_nosummary = []

def make_nosummary():
	global articles_nosummary # fix this later or something lol

	articles_nosummary = []
	for i, art in enumerate(articles):
		art_new = {}
		for key, value in art.items():
			if not key == 'summary':
				art_new[key] = value
		art_new['idx'] = i
		articles_nosummary.append(art_new)

def update(scrape_new=True):
	global articles # fix this later or something lol

	print('updating articles...')
	if scrape_new:
		print('scraping new...')
		run_engine()

	print('loading new data...')
	with open('data.json', 'rb') as file_in:
		articles = json.load(file_in)

	articles = sorted(articles, key=lambda x:x['neutrality'])
	make_nosummary()

scheduler = BackgroundScheduler()
scheduler.add_job(func=update, trigger='cron', hour='11', minute='0')
#scheduler.start()

update(scrape_new=False)

def filter(query, articles):
	results = []
	for art in articles:
		if query is None or query.lower() in art['title'].lower():
			results.append(art)
	return results

def similar(idx, articles):
	results = []

	sims = articles[idx]['sims']
	for i, prob in sims[:10]:
		results.append(articles[i])

	return results

@app.route('/summary')
def summary():
	return jsonify(articles[int(request.args.get('idx'))])

@app.route('/')
def index():
	query = request.args.get('q')
	print(query)
	if not query == None and query.startswith('document:'):
		idx = int(query[9:])
		print('document', idx)

		return render_template('index.html', articles=similar(idx, articles_nosummary))

	return render_template('index.html', articles=filter(query, articles_nosummary))

@app.route('/briefing')
def briefing():
	return render_template('briefing.html')

@app.route('/getbrief')
def getbrief():
	minutes = int(request.args.get('t'))

	print(minutes)

	curr = 0

	results = []
	article_set = set()

	counter = 0

	while curr <= minutes and counter < 20:
		for i in range(10):
			for art_idx, art in enumerate(articles):
				if not art_idx in article_set and art['max_tag'] == i and art['neutrality'] < 0.5:
					t = art['read_time']
					curr += t
					results.append(art)
					article_set.add(art_idx)
					break
		counter += 1

	return jsonify(results)

atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
	app.run(host='0.0.0.0')