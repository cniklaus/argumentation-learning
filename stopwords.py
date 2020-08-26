import spacy
from spacy.lang.de.examples import sentences
from collections import OrderedDict
import numpy as np

nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[0])


print(nlp.Defaults.stop_words)