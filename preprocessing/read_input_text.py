import spacy


nlp = spacy.load('de_core_news_sm')


def read_input(i):
    f_in = open("../Corpus/%s.txt" % i, "r")
    text = f_in.read()

    doc = nlp(text)
    #sentences = [sent.string.strip() for sent in doc.sents]

    return doc


