#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[0])
#print(doc.text)
#for token in doc:
   # print(token.text, token.pos_, token.dep_)


def sentence_closes_with_question_marks(sentence):
    if sentence[-1].text == "?":
        return True
    else:
        return False


#print(sentence_closes_with_quotation_marks(doc))