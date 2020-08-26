# people = ["Lisa","Pam","Phil","John"]
# unique pairs where order is irrelevant
# result = [print(people[p1], people[p2]) for p1 in range(len(people)) for p2 in range(p1+1,len(people))]
# All possible pairs, excluding duplicates:
# result = [print(p1, p2) for p1 in people for p2 in people if p1 != p2]

import spacy
import numpy as np
import pickle
from sklearn.svm import LinearSVC, SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataset_analysis import analyse_dataset
import csv
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from features_relations import structural_number_tokens_source, relationship, structural_difference_number_tokens, structural_sentence_distance, structural_target_before_source, lexical_number_common_terms, structural_number_punctuation_marks_source, structural_difference_number_punctuation_marks, lexical_source_contains_modal_verb, indicators_source_contains_discourse_markers

nlp = spacy.load('de_core_news_sm')

# TODO operate on clausal level!? maybe...


def create_train_test_split(df):
    y = df.loc[:, ['relationship']] # y = dataset.label
    X = df.drop(['relationship'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #random_state=1234

    return X_train, X_test, y_train, y_test


def train_linear_SVC():
    df = encode_features()
    print(df)

    classifier = LinearSVC()
    X_train, X_test, y_train, y_test = create_train_test_split(df)

    rus = RandomUnderSampler(return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train) # TODO other sampling techniques, e.g. ros = RandomOverSampler() X_ros, y_ros = ros.fit_sample(X, y)

    print('Removed indexes:', len(X_train), len(id_rus), id_rus)

    classifier.fit(X_rus, y_rus) # X_train, y_train.values.ravel()

    pickle.dump(classifier, open("linear_SVC_relations.pkl", 'wb'))

    # Predict Class
    y_pred = classifier.predict(X_test)
    print(y_pred)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("acc: ", accuracy)

    #X_unseen = create_data_set(995)

    #pred = classifier.predict(X_unseen)
    #print(pred)


def encode_features():
    f = generate_pairs()

    df = pd.DataFrame.from_dict(f)

    df = pd.get_dummies(df, columns=['label_source', 'label_target']) # 'relationship',
    # print(df['number_tokens_target'])
    scaled_features = df.copy()

    col_names = ['number_tokens_source', 'number_tokens_target', 'difference_number_tokens', 'sentence_distance', 'number_of_common_terms', 'number_of_punctuation_marks_source', 'number_of_punctuation_marks_target', 'difference_number_punctuation_marks']
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)

    scaled_features[col_names] = features
    #print(scaled_features['number_tokens_target'])

    #print(df)
    #print("df: ", df.columns.values)

    pickle.dump(scaler, open("standard_scaler_linear_SVC.pkl", 'wb'))

    print("scaled: ", scaled_features.columns.values)

    return scaled_features


def generate_features(pair1, pair2):
    features = {}
    features["relationship"] = relationship.get_relationship(pair1, pair2) # 'support' 'non-support' categorical
    features["label_source"] = pair1['label'] # claim, premise, None categorical
    features["label_target"] = pair2['label'] # claim, premise, None categorical
    features["number_tokens_source"] = structural_number_tokens_source.get_number_of_tokens(pair1['text']) # int
    features["number_tokens_target"] = structural_number_tokens_source.get_number_of_tokens(pair2['text']) # int
    features["difference_number_tokens"] = structural_difference_number_tokens.get_difference_number_tokens(pair1, pair2) # int
    features["sentence_distance"] = structural_sentence_distance.get_sentence_distance(pair1, pair2) # int
    features["target_before_source"] = structural_target_before_source.is_target_before_source(pair1, pair2) # bool
    features["number_of_common_terms"] = lexical_number_common_terms.get_number_of_common_terms(pair1, pair2) # int
    features["number_of_punctuation_marks_source"] = structural_number_punctuation_marks_source.get_number_of_punctuation_marks(pair1['text']) # int
    features["number_of_punctuation_marks_target"] = structural_number_punctuation_marks_source.get_number_of_punctuation_marks(pair2['text']) # int
    features["difference_number_punctuation_marks"] = structural_difference_number_punctuation_marks.get_difference_number_of_punctuation_marks(pair1, pair2) # int
    features["contains_modal_verb_source"] = lexical_source_contains_modal_verb.contains_modal_verb(pair1['text']) # bool
    features["contains_modal_verb_target"] = lexical_source_contains_modal_verb.contains_modal_verb(pair2['text']) # bool
    features["contains_argumentative_discourse_markers_source"] = indicators_source_contains_discourse_markers.contains_argumentative_markers(pair1['text']) # bool
    features["contains_argumentative_discourse_markers_target"] = indicators_source_contains_discourse_markers.contains_argumentative_markers(pair2['text']) # bool
    return features


def generate_pairs():
    f = []
    features = {}

    annotations = create_csv(True)

    counter = 0

    number_of_docs = annotations[-1]['docID'] + 1
    #print(number_of_docs)

    while counter < number_of_docs:
        to_compare = []
        for a in annotations:
            if a['docID'] == counter:
                to_compare.append(a)

        #print(to_compare)

        n_pair1 = 0

        while n_pair1 < len(to_compare)-1:
            n_pair2 = 0 #n_pair1 + 1
            while n_pair2 < len(to_compare):
                pair1 = to_compare[n_pair1]
                pair2 = to_compare[n_pair2]
                print("1: ", pair1)
                print("2: ", pair2)

                feat = generate_features(pair1, pair2)
                f.append(feat)

                n_pair2 += 1

            n_pair1 += 1

        counter += 1

    #print(f)

    with open('sentence_level_annotations_with_relations2.csv', mode='w') as csv_file:
        fieldnames = ['relationship', 'label_source', 'label_target', 'number_tokens_source', 'number_tokens_target', 'difference_number_tokens', 'sentence_distance', 'target_before_source', 'number_of_common_terms', 'number_of_punctuation_marks_source', 'number_of_punctuation_marks_target', 'difference_number_punctuation_marks', 'contains_modal_verb_source', 'contains_modal_verb_target', 'contains_argumentative_discourse_markers_source', 'contains_argumentative_discourse_markers_target']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for entry in f:
            writer.writerow(entry)

    return f

    # people = ["Lisa", "Pam", "Phil", "John"]
    # unique pairs where order is irrelevant
    # result = [print(people[p1], people[p2]) for p1 in range(len(people)) for p2 in range(p1 + 1, len(people))]

    # All possible pairs, excluding duplicates:

   # pairs = [(p1, p2) for p1 in sentences for p2 in sentences if p1 != p2]
   # return pairs


def create_csv(three_classes):
    f = []

    global_id = 0
    doc_id = 0

    final_annot = []

    for i in range(0,990):
        doc_annotations = []
        f_in = open("../Corpus/%s.txt" % i, "r")
        text = f_in.read()

        doc = nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]

        f_ann = open("../Corpus/%s.ann" % i, "r")
        line = f_ann.readline()

        annotated_sentences = []
        related_sentences = []
        while line:
            supports = 'None'
            l = line.split("\t")
            if l[0].startswith("T"):
                annot_id = l[0]
                annot_text = l[2].strip()
                label = l[1].split(" ")[0]
                if not three_classes:
                    if label in ('Claim', 'Premise'):
                        label = 'Argumentative'

                annotated_sentences.append({
                    'annotID': annot_id,
                    'annot_text': annot_text,
                    'label': label,
                    'supports': supports,
                })

            elif l[0].startswith("R"):
                arg1 = l[1].split(" ")[1].split(":")[1]
                arg2 = l[1].split(" ")[2].split(":")[1]

                related_sentences.append({
                    'arg1': arg1,
                    'arg2': arg2,
                })

            line = f_ann.readline()

        for r in related_sentences:
            for a in annotated_sentences:
                if r['arg1'] == a['annotID']:
                    a['supports'] = r['arg2']

        for s in sentences:
            annotated = False
            for a in annotated_sentences:
                if a['annot_text'] in s:
                    annotated = True
                    final_annot.append({
                        'globalID': global_id,
                        'docID': doc_id,
                        'annotID': a['annotID'],
                        'text': s.strip(),
                        'label': a['label'],
                        'supports': a['supports'],
                    })
                    doc_annotations.append({
                        'globalID': global_id,
                        'docID': doc_id,
                        'annotID': a['annotID'],
                        'text': s.strip(),
                        'label': a['label'],
                        'supports': a['supports'],
                    })
            if not annotated:
                final_annot.append({
                    'globalID': global_id,
                    'docID': doc_id,
                    'annotID': 'None',
                    'text': s.strip(),
                    'label': 'None',
                    'supports': 'None',
                })
                doc_annotations.append({
                    'globalID': global_id,
                    'docID': doc_id,
                    'annotID': 'None',
                    'text': s.strip(),
                    'label': 'None',
                    'supports': 'None',
                })
            global_id += 1

        doc_id += 1

    with open('sentence_level_annotations_with_relations.csv', mode='w') as csv_file:
        fieldnames = ['globalID', 'docID', 'annotID', 'text', 'label', 'supports']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for f in final_annot:
            writer.writerow(f)

    return final_annot


#create_csv(True)
#generate_pairs()
#encode_features()
train_linear_SVC()