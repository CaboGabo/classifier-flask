from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from classifiers import *

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/diagnosticate', methods=['POST'])
def diagnosticate():
    data = request.get_json()
    posts = data['posts']

    posts = getTokenizedText(posts)
    total_words = []

    results = []
    for post in posts:
        post['content'] = removeWords(post['content'])
        total_words += post['content']
        [stemmed]  = stemming([post])
        post = stemmed
        postTags = []
        for classifier in classifiers:
            post_document = documentFeatures(post, classifier[1])
            postTags.append(classifier[0].classify(post_document))

        output = {
            "post": post,
            "tags": postTags
        }
        results.append(output)

    response = {}
    all_words = FreqDist(total_words)
    words_used = list(all_words.keys())[:10]
    response['classifiedPosts'] = results
    response['topWords'] = words_used

    return jsonify(response)