import nltk
from glob import glob
from nltk.stem import PorterStemmer ,WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
files = glob('dataset/trainData/DavidLawder/*')
for ifile in files:
    file = open(ifile, 'r')
    lines = file.readlines()
    # print(lines)
    # print('=================================================================================================================================================')
    for line in lines:
        for sentence in nltk.sent_tokenize(line):
            words = nltk.word_tokenize(sentence)
            for i in range(len(words)):
                words[i] = porter.stem(words[i])
            print(' '.join(words).lower())


# print(lemmatizer.lemmatize('better'))
# print(porter.stem('byed'))
