import spacy
from preprocessing import read_input_text

nlp = spacy.load('de_core_news_sm')


def get_number_of_common_terms(pair1, pair2):
    token_list1 = set([token.text.lower() for token in nlp(pair1['text'])])
    token_list2 = set([token.text.lower() for token in nlp(pair2['text'])])

    counter = 0
    for t in token_list1:
        if t in token_list2:
            counter += 1

    return counter


# p1 = {'text': "Das wetter ist schön."}
# p2 = {'text': "Das Wetter ist schlecht."}
# print(get_number_of_common_terms(p1, p2))