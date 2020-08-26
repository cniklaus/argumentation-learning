import spacy
from syntactic_features import pos, ner, dependency
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from syntactic_features import ner
from sklearn.model_selection import train_test_split
from structural_features import sentence_closes_with_question_mark, sentence_length
from lexical_features import contains_adverb, contains_modal_verb
from indicators import reference_to_first_person, argumentative_discourse_markers
from dataset_analysis import analyse_dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from classifiers_components import dataframe
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer


nlp = spacy.load('de_core_news_sm')

# TODO operate on clausal level!


def get_label(sentence, annotations):
    for ann in annotations:
        if ann[0] in sentence:
            assigned_label = ann[1]
            return assigned_label
    return "None"


def create_dataframe(f):
    df = pd.DataFrame.from_dict(f)

    cv = CountVectorizer(max_features=1000, ngram_range=(1, 3)) #, min_df=0.1, max_df=0.7


    X1 = cv.fit_transform(df.text)
    #print(X1.toarray())
    #print(cv.get_feature_names())
    df = df.drop(columns='text')
    count_vect_df = pd.DataFrame(X1.todense(), columns=cv.get_feature_names())
    #print(pd.concat([df, count_vect_df], axis=1))

    #X1 = mlb_ner.fit_transform(df.ner)
    #df = df.drop(columns='ner')
    #ner_df = df.join(pd.DataFrame(X1, columns=mlb_ner.classes_))

    #X1 = mlb_pos.fit_transform(df.pos)
    #df = df.drop(columns='pos')
    #pos_df = df.join(pd.DataFrame(X1, columns=mlb_pos.classes_))

    #X1 = mlb_dep.fit_transform(df.dep)
    #df = df.drop(columns='dep')
    #dep_df = df.join(pd.DataFrame(X1, columns=mlb_dep.classes_))

    ####X = mlb.fit_transform(df.label)
    ####df = df.drop(columns='label')
    ####df = df.join(pd.DataFrame(X, columns=mlb.classes_))

    ####print(df.to_string())

    combined_df = pd.concat([df, count_vect_df], axis=1)

    #pickle.dump(cv, open('GaussianNB/cv.pkl', 'wb'))

    return combined_df, cv


def create_data_set(three_classes):
    f = []
    for i in range(0,990):
        f_in = open("../Corpus/%s.txt" % i, "r")
        text = f_in.read()

        f_ann = open("../Corpus/%s.ann" % i, "r")
        line = f_ann.readline()
        annotations = []
        while line:
            l = line.split("\t")
            if l[0].startswith("T"):
                annotation = l[2]
                label = l[1].split(" ")[0]

                if not three_classes:
                    if label in ('Claim', 'Premise'):
                        label = 'Argumentative'
                annotations.append((annotation.strip(), label))
            line = f_ann.readline()

        #for a in annotations:
            #print(a)

        doc = nlp(text)

        sentences = [sent.string.strip() for sent in doc.sents]

        for s in sentences:
            features = {}
            sen = nlp(s)
            features["label"] = get_label(s, annotations)
            features["text"] = s
            #features["pos"] = pos.get_pos_tags(sen)
            #features["dep"] = dependency.get_dep_tags(sen)
            #features["ner"] = ner.get_ner(sen)
            features["closing_question_mark"] = sentence_closes_with_question_mark.sentence_closes_with_question_marks(sen) # bool
            features["contains_adverb"] = contains_adverb.contains_adverb(sen) # bool
            features["contains_modal"] = contains_modal_verb.contains_modal_verb(sen) # bool
            features["first_person"] = reference_to_first_person.contains_first_person(sen) # bool
            #features["causal_markers"] = argumentative_discourse_markers.contains_causal_markers(sen) # bool
            #features["conditional_markers"] = argumentative_discourse_markers.contains_conditional_markers(sen) # bool
            #features["adversative_markers"] = argumentative_discourse_markers.contains_adversative_markers(sen) # bool
            #features["consecutive_markers"] = argumentative_discourse_markers.contains_consecutive_markers(sen) # bool
            #features["concessive_markers"] = argumentative_discourse_markers.contains_concessive_markers(sen) # bool
            features["argumentative_discourse_markers"] = argumentative_discourse_markers.contains_argumentative_markers(sen)
            features["contains_named_entities"] = ner.contains_ner(sen)
            features["sentence_length"] = sentence_length.get_sentence_length(sen)
            f.append(features)
            #print(features)

    #print(f)

    dataset = create_dataframe(f)

    return dataset


def create_train_test_split():
    dataset, cv = create_data_set(True)
    # display the dimensions of the dataset

    analyse_dataset.analyse(dataset)

    # save DataFrame
    dataset.to_csv('analytical_base_table.csv', index=None)

    y = dataset.loc[:, ['label']] # y = dataset.label
    X = dataset.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #random_state=1234

    return cv, X_train, X_test, y_train, y_test



