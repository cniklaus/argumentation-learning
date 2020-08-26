#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[1])
#print(doc.text)
#for token in doc:
    #print(token.text, token.pos_, token.dep_)


def contains_adverb(sentence):
    pos_list = [token.pos_ for token in sentence]

    for p in pos_list: #TODO use tag_ instead of pos_???
        if p == 'ADV':
            return True

    return False
