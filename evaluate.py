import nltk
from glob import glob
import kenlm
import math
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
porter = PorterStemmer()

models = [
    kenlm.LanguageModel('models/AaronPressman.arpa'),
    kenlm.LanguageModel('models/AlanCrosby.arpa'),
    kenlm.LanguageModel('models/AlexanderSmith.arpa'),
    kenlm.LanguageModel('models/BenjaminKangLim.arpa'),
    kenlm.LanguageModel('models/BernardHickey.arpa'),
    kenlm.LanguageModel('models/BradDorfman.arpa'),
    kenlm.LanguageModel('models/DarrenSchuettler.arpa'),
    kenlm.LanguageModel('models/DavidLawder.arpa')
]
authors = ['AaronPressman', 'AlanCrosby', 'AlexanderSmith', 'BenjaminKangLim', 'BernardHickey', 'BradDorfman', 'DarrenSchuettler', 'DavidLawder']
print('______________________________________________________________________________________________________________')

def stemming(sentence):
    words = nltk.word_tokenize(sentence)
    for i in range(len(words)):
        words[i] = porter.stem(words[i])
    sentence = ' '.join(words)
    return sentence

initial_values = [[0 for i in range(8)] for j in range(8)]
conf_matrix = pd.DataFrame(initial_values, index=authors, columns=authors)

dirs = glob('dataset/testPR_Recall/*')
for idir in dirs:
    files = glob(idir+'/*')
    real_author = authors.index(idir[22:])
    for ifile in files:
        mini = math.inf
        pred_author = ''
        file = open(ifile, 'r')
        sent = file.read()
        sent = stemming(sent)
        for i in range(len(models)):
            model = models[i]
            sum_inv_logs = -1 * sum(score for score, _, _ in model.full_scores(sent))
            n = len(list(model.full_scores(sent)))
            perplexity = math.pow(10.0, sum_inv_logs / n)
            if perplexity < mini:
                pred_author = i
                mini = perplexity
        # print(pred_author)
        conf_matrix[authors[real_author]][authors[pred_author]] += 1

TPs = [0 for i in range(8)]
FPs = [0 for i in range(8)]
FNs = [0 for i in range(8)]
precisions = [0 for i in range(8)]
recalls = [0 for i in range(8)]
for real in authors:
    for pred in authors:
        if real == pred:
            index = authors.index(real)
            TPs[index] += conf_matrix[pred][real]
        else:
            index = authors.index(real)
            FNs[index] += conf_matrix[pred][real]
for real in authors:
    for pred in authors:
        if real != pred:
            index = authors.index(real)
            FPs[index] += conf_matrix[real][pred]
for i in range(8):
    precisions[i] = TPs[i]/(TPs[i]+FPs[i])
    recalls[i] = TPs[i]/(TPs[i]+FNs[i])
precisions_mean = str(np.mean(precisions))
recalls_mean = str(np.mean(recalls))
print('precision = ', precisions_mean, ' | recall = ', recalls_mean)

