#import spacy
#from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

#nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[1])

causal_markers = ["da", "denn", "weil"]
consecutive_markers = ["also", "dadurch", "daher", "darum", "deshalb", "deswegen", "drum"]
adversative_markers = ["anstatt", "anderenfalls", "andererseits", "einerseits", "hingegen", "jedoch", "sondern", "statt", "stattdessen", "vielmehr", "wiederum", "zum einen", "zum anderen"]
concessive_markers = ["allerdings", "dennoch", "gleichwohl", "nichtsdestotrotz", "nichtsdestoweniger", "obschon", "obwohl", "trotzdem", "wenngleich", "wobei", "zwar"]
conditional_markers = ["falls", "ob", "sofern", "wenn"]

argumentative_discourse_markers = causal_markers + consecutive_markers + adversative_markers + concessive_markers + conditional_markers


def contains_causal_markers(sentence):
    for token in sentence:
        if token.text.lower() in causal_markers:
            return True
    return False


def contains_consecutive_markers(sentence):
    for token in sentence:
        if token.text.lower() in consecutive_markers:
            return True
    return False


def contains_adversative_markers(sentence):
    for token in sentence:
        if token.text.lower() in adversative_markers:
            return True
    return False


def contains_concessive_markers(sentence):
    for token in sentence:
        if token.text.lower() in concessive_markers:
            return True
    return False


def contains_conditional_markers(sentence):
    for token in sentence:
        if token.text.lower() in conditional_markers:
            return True
    return False


def contains_argumentative_markers(sentence):
    argumentative_discourse_markers = causal_markers + consecutive_markers + adversative_markers + concessive_markers + conditional_markers
    for token in sentence:
        if token.text.lower() in argumentative_discourse_markers:
            return True
    return False


