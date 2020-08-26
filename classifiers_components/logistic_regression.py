from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from classifiers_components import dataframe
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#X_train, X_test, y_train, y_test = dataframe.create_train_test_split()

#logisticRegr = LogisticRegression()
#logisticRegr.fit(X_train, y_train.values.ravel())
#y_pred3 = logisticRegr.predict(X_test)
#print(y_pred3)
#accuracy3 = accuracy_score(y_test, y_pred3)
#print("logistic regression: ", accuracy3)

import spacy
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from classifiers_components import dataframe
import pickle
import pandas as pd
from indicators import argumentative_discourse_markers, reference_to_first_person
from structural_features import sentence_closes_with_question_mark, sentence_length
from lexical_features import contains_modal_verb, contains_adverb
from syntactic_features import ner
import numpy as np
from sklearn.model_selection import GridSearchCV


nlp = spacy.load('de_core_news_sm')


cv, X_train, X_test, y_train, y_test = dataframe.create_train_test_split()





# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)




classifier = LogisticRegression()


clf = GridSearchCV(classifier, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(X_train, y_train.values.ravel())

#classifier.fit(X_train, y_train.values.ravel())

#pickle.dump(classifier, open("Gaussian_NB_model.pkl", 'wb'))

# Predict Class
y_pred = best_model.predict(X_test)
print(y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("NB: ", accuracy)


def create_data_set(i):
    f = []

    f_in = open("../Corpus/%s.txt" % i, "r")
    text = f_in.read()

    doc = nlp(text)

    sentences = [sent.string.strip() for sent in doc.sents]

    for s in sentences:
        print(s)
        features = {}
        sen = nlp(s)

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

    df = pd.DataFrame.from_dict(f)

    X1 = cv.transform(df.text)
    # print(X1.toarray())
    # print(cv.get_feature_names())
    df = df.drop(columns='text')
    count_vect_df = pd.DataFrame(X1.todense(), columns=cv.get_feature_names())
    # print(pd.concat([df, count_vect_df], axis=1))

    combined_df = pd.concat([df, count_vect_df], axis=1)

    return combined_df


X_unseen = create_data_set(995)

pred = best_model.predict(X_unseen)
print(pred)



