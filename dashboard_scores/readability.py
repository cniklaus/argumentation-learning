import textstat
from preprocessing import read_input_text
import spacy
import pyphen
from dashboard_scores import basic_scores

nlp = spacy.load('de_core_news_sm')
dic = pyphen.Pyphen(lang='de')


# print(dic.inserted('Apfelbaum'))


def count_syllables(token):
    split_token = dic.inserted(token)
    syllables = split_token.split("-")
    return len(syllables)


def FRE_German(text):
    count_s = 0
    for token in text:
        count_s = count_s + count_syllables(token.text)

    number_of_sentences = basic_scores.get_number_of_sentences(text)
    tokens = [token.text for token in text]

    asw = count_s / len(tokens)
    asl = len(tokens) / number_of_sentences

    fre = 180 - asl - (58.5 * asw)

    return fre


def Wiener_Sachtextformel_1(text):
    number_of_sentences = basic_scores.get_number_of_sentences(text)
    tokens = [token.text for token in text]
    count_s = 0
    one_syllable_word = 0
    three_or_more_syllable_word = 0
    more_than_six_characters_word = 0

    for token in text:
        if len(token) > 6:
            more_than_six_characters_word += 1

        number_syllables = count_syllables(token.text)

        if number_syllables == 1:
            one_syllable_word += 1
        elif number_syllables > 2:
            three_or_more_syllable_word += 1

        count_s = count_s + count_syllables(token.text)

    ms = (three_or_more_syllable_word * 100) / len(tokens)
    sl = len(tokens) / number_of_sentences
    iw = (more_than_six_characters_word * 100) / len(tokens)
    es = (one_syllable_word * 100) / len(tokens)

    wstf1 = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875

    return wstf1


# TODO: check which need to be adapated for being applicable on German texts!!!


def get_flesch_reading_ease(text):
    return textstat.flesch_reading_ease(text.text)


def get_smog_index(text):
    return textstat.smog_index(text.text)


def get_flesch_kincaid_grade(text):
    return textstat.flesch_kincaid_grade(text.text)


def get_coleman_liau_index(text):
    return textstat.coleman_liau_index(text.text)


def get_automated_readability_index(text):
    return textstat.automated_readability_index(text.text)


def get_dale_chall_readability_score(text):
    return textstat.dale_chall_readability_score(text.text)


def get_difficult_words(text):
    return textstat.difficult_words(text.text)


def get_linsear_write_formula(text):
    return textstat.linsear_write_formula(text.text)


def get_gunning_fog(text):
    return textstat.gunning_fog(text.text)


def get_text_standard(text):
    return textstat.text_standard(text.text)


#print(get_flesch_reading_ease(read_input_text.read_input(2)))
#print(get_smog_index(read_input_text.read_input(2)))
#print(get_flesch_kincaid_grade(read_input_text.read_input(2)))
#print(get_coleman_liau_index(read_input_text.read_input(2)))
#print(get_automated_readability_index(read_input_text.read_input(2)))
#print(get_dale_chall_readability_score(read_input_text.read_input(2)))
#print(get_difficult_words(read_input_text.read_input(2)))
#print(get_linsear_write_formula(read_input_text.read_input(2)))
#print(get_gunning_fog(read_input_text.read_input(2)))
#print(get_text_standard(read_input_text.read_input(2)))


