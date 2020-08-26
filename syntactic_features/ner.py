#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[0])
#print(doc.text)
#for token in doc:
    #print(token.text, token.pos_, token.dep_)


def get_ner(sentence):
    #for ent in sentence.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)

    ner_list = [ent.label_ for ent in sentence.ents]
    return ner_list


def contains_ner(sentence):
    ner_list = get_ner(sentence)

    if len(ner_list) > 0:
        return True

    return False


#print(get_ner(doc))