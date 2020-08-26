import spacy

nlp = spacy.load('de_core_news_sm')


def contains_modal_verb(sentence):
    pos_list = [token.tag_ for token in nlp(sentence)]

    for p in pos_list:
        if p in ('VMFIN', 'VMINF', 'VMPP'):
            return True

    return False


# print(contains_modal_verb("Ich würde dies tun."))