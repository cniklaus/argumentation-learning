import spacy

nlp = spacy.load('de_core_news_sm')


def get_number_of_punctuation_marks(sentence):
    tokens = [token.pos_ for token in nlp(sentence)]

    counter = 0
    for t in tokens:
        if t == 'PUNCT':
            counter += 1

    return counter


# print(get_number_of_punctuation_marks("Das Wetter, welches ich liebe, ist schön."))