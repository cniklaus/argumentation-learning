import spacy
from preprocessing import read_input_text

nlp = spacy.load('de_core_news_sm')


def get_number_of_sentences(text):
    sentences = [sent.string.strip() for sent in text.sents]

    return len(sentences)


def get_average_sentence_length(text):
    number_of_sentences = get_number_of_sentences(text)
    tokens = [token.text for token in text]

    return len(tokens) / number_of_sentences


#print(get_number_of_sentences(read_input_text.read_input(11)))
#print(get_average_sentence_length(read_input_text.read_input(11)))
