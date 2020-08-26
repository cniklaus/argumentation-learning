import spacy
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from classifiers_components import dataframe
import pickle
import pandas as pd
from indicators import argumentative_discourse_markers, reference_to_first_person
from structural_features import sentence_closes_with_question_mark, sentence_length
from lexical_features import contains_modal_verb, contains_adverb
from syntactic_features import ner
from sklearn.model_selection import GridSearchCV

nlp = spacy.load('de_core_news_sm')

# Predict Class
# y_pred = classifier.predict(X_test)
# print(y_pred)

# Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("NB: ", accuracy)


def create_data_set(text, cv):
    f = []

    # read text
    #f_in = open("../Corpus/%s.txt" % i, "r")
    #text = f_in.read()

    doc = nlp(text)

    sentences = [sent.string.strip() for sent in doc.sents]

    # features
    for s in sentences:
        #print(s)
        features = {}
        sen = nlp(s)

        features["text"] = s
        features["closing_question_mark"] = sentence_closes_with_question_mark.sentence_closes_with_question_marks(sen) # bool
        features["contains_adverb"] = contains_adverb.contains_adverb(sen) # bool
        features["contains_modal"] = contains_modal_verb.contains_modal_verb(sen) # bool
        features["first_person"] = reference_to_first_person.contains_first_person(sen) # bool
        features["argumentative_discourse_markers"] = argumentative_discourse_markers.contains_argumentative_markers(sen)
        features["contains_named_entities"] = ner.contains_ner(sen)
        features["sentence_length"] = sentence_length.get_sentence_length(sen)
        f.append(features)

    df = pd.DataFrame.from_dict(f)

    X1 = cv.transform(df.text)
    df = df.drop(columns='text')
    count_vect_df = pd.DataFrame(X1.todense(), columns=cv.get_feature_names())

    combined_df = pd.concat([df, count_vect_df], axis=1)

    return combined_df


def predict(text):
    # load the model from disk
    loaded_model = pickle.load(open("linear_SVC.pkl", 'rb'))
    loaded_count_vectorizer = pickle.load(open("count_vectorizer_linear_SVC.pkl", 'rb'))

    X_unseen = create_data_set(text, loaded_count_vectorizer)

    pred = loaded_model.predict(X_unseen)
    #print(pred)
    return pred


#i = 995
#f_in = open("../Corpus/%s.txt" % i, "r")
#text = f_in.read()
#predict(text)