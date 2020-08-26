#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[0])
#print(doc.text)
#for token in doc:
    #print(token.text, token.pos_, token.dep_)


def contains_first_person(sentence):
    token_list = [token.text.lower() for token in sentence]

    for t in token_list:
        if t in ("ich", "mich", "mir", "mein", "meine", "meiner", "meinen", "meinem", "meines"):
            return True

    return False


#print(contains_first_person(nlp("Mein Hund ist groß.")))