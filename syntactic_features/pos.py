#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[0])
#print(doc.text)
#for token in doc:
    #print(token.text, token.pos_, token.dep_)


def get_pos_tags(sentence):
    pos_list = [token.pos_ for token in sentence]
    return pos_list


