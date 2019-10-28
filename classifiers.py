import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
import random
import math
import nltk

# Código que carga los json, y los lee
a2 = open('datasets/datasetA2.json', 'r', encoding='utf8')
a3 = open('datasets/datasetA3.json', 'r', encoding='utf8')
a4 = open('datasets/datasetA4.json', 'r', encoding='utf8')
a6 = open('datasets/datasetA6.json', 'r', encoding='utf8')
a7 = open('datasets/datasetA7.json', 'r', encoding='utf8')
a8 = open('datasets/datasetA8.json', 'r', encoding='utf8')
a9 = open('datasets/datasetA9.json', 'r', encoding='utf8')
b1 = open('datasets/datasetB1.json', 'r', encoding='utf8')
b4 = open('datasets/datasetB4.json', 'r', encoding='utf8')
b6 = open('datasets/datasetB6.json', 'r', encoding='utf8')
c1 = open('datasets/datasetC1.json', 'r', encoding='utf8')

texta2 = a2.read()
texta3 = a3.read()
texta4 = a4.read()
texta6 = a6.read()
texta7 = a7.read()
texta8 = a8.read()
texta9 = a9.read()
textb1 = b1.read()
textb4 = b4.read()
textb6 = b6.read()
textc1 = c1.read()

a2.close()
a3.close()
a4.close()
a6.close()
a7.close()
a8.close()
a9.close()
b1.close()
b4.close()
b6.close()
c1.close()

objsa2 = json.loads(texta2)
objsa3 = json.loads(texta3)
objsa4 = json.loads(texta4)
objsa6 = json.loads(texta6)
objsa7 = json.loads(texta7)
objsa8 = json.loads(texta8)
objsa9 = json.loads(texta9)
objsb1 = json.loads(textb1)
objsb4 = json.loads(textb4)
objsb6 = json.loads(textb6)
objsc1 = json.loads(textc1)

# Función que quita las palabras que no son relevantes dentro de las frases
def removeWords(tokenized_word):
    stop_words = set(stopwords.words('spanish'))
    filtered_sent = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent


# Función que obtiene la cadena tokenizada
def getTokenizedText(arrayObj):
    for obj in arrayObj:
        tokenized_text = word_tokenize(obj['content'])
        obj['content'] = removeWords(tokenized_text)
    return arrayObj

# Función que obtiene todas las palabras usadas en el conjunto de entrenamiento


def getAllWords(arrayObjTokenized):
    words = []
    for obj in arrayObjTokenized:
        for w in obj['content']:
            words.append(w.lower())
    return words

# Función que acomoda la info para ser procesada


def documentFeatures(obj, word_features):
    features = {}
    [objStemmed] = stemming([obj])
    document_words = set(objStemmed['content'])
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    features['sentimen'] = obj['sentiment']
    features['magnitude'] = obj['magnitude']
    return features

# Función que entrena el clasificador


def stemming(objArray):
    stemmer = SnowballStemmer('spanish')
    for obj in objArray:
        stemmed_words = []
        for w in obj['content']:
            stemmed_words.append(stemmer.stem(w))

        obj['content'] = stemmed_words

    return objArray


def getClassifier(objArray):
    random.shuffle(objArray)
    objArray = stemming(getTokenizedText(objArray))
    all_words = FreqDist(getAllWords(objArray))

    word_features = list(all_words.keys())[:200]

    featuresets = []
    for document in objArray:
        featuresets += [(documentFeatures(document,
                                          word_features), document['tag'])]

    train_set_len = math.floor(len(featuresets)*0.1)
    train_set, test_set = featuresets[train_set_len:], featuresets[:train_set_len]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(classifier, test_set))
    return [classifier, word_features]

classifiers = [getClassifier(objsa2), getClassifier(objsa3), getClassifier(objsa4), getClassifier(objsa6), getClassifier(objsa7), getClassifier(
    objsa8), getClassifier(objsa9), getClassifier(objsb1), getClassifier(objsb4), getClassifier(objsb6), getClassifier(objsc1)]

print('Clasificadores entrenados')