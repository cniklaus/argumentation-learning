import spacy
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from structural_features import sentence_closes_with_question_mark
from lexical_features import contains_adverb, contains_modal_verb
from indicators import reference_to_first_person, argumentative_discourse_markers
from syntactic_features import pos, ner, dependency

nlp = spacy.load('de_core_news_sm')


def read_input(i):
    f_in = open("Corpus/%s.txt" % i, "r")
    text = f_in.read()

    doc = nlp(text)
    #sentences = [sent.string.strip() for sent in doc.sents]
    return doc


def create_dataframe(f):
    df = pd.DataFrame.from_dict(f)

    mlb = MultiLabelBinarizer()
    # unigrams to trigrams
    cv = CountVectorizer(max_features=1000, ngram_range=(1, 3))  # , min_df=0.1, max_df=0.7

    X1 = cv.transform(df.text)
    df = df.drop(columns='text')
    count_vect_df = pd.DataFrame(X1.todense(), columns=cv.get_feature_names())

    X1 = mlb.transform(df.ner)
    df = df.drop(columns='ner')
    df = df.join(pd.DataFrame(X1, columns=mlb.classes_))

    X1 = mlb.transform(df.pos)
    df = df.drop(columns='pos')
    df = df.join(pd.DataFrame(X1, columns=mlb.classes_))

    X1 = mlb.transform(df.dep)
    df = df.drop(columns='dep')
    df = df.join(pd.DataFrame(X1, columns=mlb.classes_))

    combined_df = pd.concat([df, count_vect_df], axis=1)

    return combined_df


def create_data_set(three_classes, i):
    f = []
    doc = read_input(i)
    sentences = [sent.string.strip() for sent in doc.sents]

    print(sentences)

    for s in sentences:
        features = {}
        sen = nlp(s)
        features["text"] = s
        features["pos"] = pos.get_pos_tags(sen)
        features["dep"] = dependency.get_dep_tags(sen)
        features["ner"] = ner.get_ner(sen)
        features["closing_question_mark"] = sentence_closes_with_question_mark.sentence_closes_with_question_marks(sen)  # bool
        features["contains_adverb"] = contains_adverb.contains_adverb(sen)  # bool
        features["contains_modal"] = contains_modal_verb.contains_modal_verb(sen)  # bool
        features["first_person"] = reference_to_first_person.contains_first_person(sen)  # bool
        features["argumentative_discourse_markers"] = argumentative_discourse_markers.contains_argumentative_markers(sen)
        f.append(features)
        # print(features)

    # print(f)

    dataset = create_dataframe(f)

    return dataset


