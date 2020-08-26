import spacy

nlp = spacy.load('de_core_news_sm')


def get_number_of_tokens(sentence):
    tokens = [token.text for token in nlp(sentence)]

    return len(tokens)


# print(get_number_of_tokens("Das Wetter ist schön."))