from flask import Flask, jsonify, request
from classifiers import *

app = Flask(__name__)

@app.route('/diagnosticate', methods=['POST'])
def diagnosticate():
    classifiers = [getClassifier(objsa2), getClassifier(objsa3), getClassifier(objsa4), getClassifier(objsa6), getClassifier(objsa7), getClassifier(
    objsa8), getClassifier(objsa9), getClassifier(objsb1), getClassifier(objsb4), getClassifier(objsb6), getClassifier(objsc1)]
    data = request.get_json()
    posts = data['posts']

    posts = getTokenizedText(posts)
    total_words = []

    results = []
    for post in posts:
        post['text'] = removeWords(post['text'])
        total_words += post['text']
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