import nltk
from glob import glob
import kenlm
import math
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

outfile = open("output.txt", "a")
files = glob('dataset/testData/*')
for ifile in files:
    mini = math.inf
    file_name = ifile[17:-4]
    file = open(ifile, 'r')
    sent = file.read()
    sent = stemming(sent)
    for i in range(len(models)):
        model = models[i]
        sum_inv_logs = -1 * sum(score for score, _, _ in model.full_scores(sent))
        n = len(list(model.full_scores(sent)))
        perplexity = math.pow(10.0, sum_inv_logs / n)
        if perplexity < mini:
            pred_author = authors[i]
            mini = perplexity
    outfile.write(file_name+"   "+pred_author+'\n')
outfile.close()
