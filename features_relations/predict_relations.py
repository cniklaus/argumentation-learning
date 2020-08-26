import spacy
import pickle
from sklearn.svm import LinearSVC, SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataset_analysis import analyse_dataset
import csv
from sklearn.metrics import accuracy_score
from features_relations import structural_number_tokens_source, relationship, structural_difference_number_tokens, structural_sentence_distance, structural_target_before_source, lexical_number_common_terms, structural_number_punctuation_marks_source, structural_difference_number_punctuation_marks, lexical_source_contains_modal_verb, indicators_source_contains_discourse_markers

nlp = spacy.load('de_core_news_sm')

# TODO Reihenfolge der pairs relevant?!
# TODO insert None-label!!! as src and target!!!

elements = []

elements.append({
                'id': 0,
                'label': 'claim',
                'start': 0,
                'length': 20,
                'text': "Ich finde das Produkt gut.",
                'confidence': 1.
            })


elements.append({
                'id': 1,
                'label': 'premise',
                'start': 40,
                'length': 60,
                'text': "Meiner Meinung nach ist es äußerst hilfreich.",
                'confidence': 1.
            })

# elements.append({
  #              'id': 2,
  #              'label': 'None',
  #              'start': 97,
  #              'length': 100,
  #              'text': "Das Wetter ist schön.",
  #              'confidence': 1.
  #          })

elements.append({
                'id': 2,
                'label': 'premise',
                'start': 70,
                'length': 75,
                'text': "Es löst das gegebene Problem.",
                'confidence': 1.
            })

# elements.append({
#              'id': 4,
#                'label': 'None',
#                'start': 97,
#                'length': 100,
#                'text': "Das Wetter ist schön.",
#                'confidence': 1.
#            })


def predict(elements):
    # load the model from disk
    loaded_model = pickle.load(open("linear_SVC_relations.pkl", 'rb'))

    X_unseen_pairs, pairs = encode_features(elements)
    print("pairs: ", X_unseen_pairs)
    #for x in X_unseen_pairs:
     #   pred = loaded_model.predict(x)
      #  print(pred)
    # return pred
    pred = loaded_model.predict(X_unseen_pairs)
    support_relations = []

    count = 0
    for p in pred:
        print(p)
        if p == 1 and pairs[count]['srcElem'] != pairs[count]['trgElem']:
            pairs[count]['label'] = 'support'
        count += 1
    print(pred)

    for p in pairs:
        if p['label'] == 'support':
            support_relations.append(p)

    return support_relations


def encode_features(elements):
    f, pairs = generate_pairs(elements)

    # loaded_model = pickle.load(open("linear_SVC.pkl", 'rb'))
    loaded_scaler = pickle.load(open("standard_scaler_linear_SVC.pkl", 'rb'))

    df = pd.DataFrame.from_dict(f)

    df = pd.get_dummies(df, columns=['label_source', 'label_target']) # 'relationship',
    # print(df['number_tokens_target'])
    scaled_features = df.copy()

    col_names = ['number_tokens_source', 'number_tokens_target', 'difference_number_tokens', 'sentence_distance', 'number_of_common_terms', 'number_of_punctuation_marks_source', 'number_of_punctuation_marks_target', 'difference_number_punctuation_marks']
    features = scaled_features[col_names]
    # scaler = StandardScaler().fit(features.values)
    features = loaded_scaler.transform(features.values)

    scaled_features[col_names] = features
    #print(scaled_features['number_tokens_target'])

    #print(df)
    #print("df: ", df.columns.values)

    print("scaled: ", scaled_features.columns.values)
    print(scaled_features)

    scaled_features["label_source_None"] = 0
    scaled_features["label_target_None"] = 0

    return scaled_features, pairs


def generate_pairs(elements):
    f = []
    annotations = create_csv(elements)
    counter = 0
    number_of_docs = 1
    #print(number_of_docs)
    pairs = []

    while counter < number_of_docs:
        to_compare = []
        for a in annotations:
            # if a['docID'] == counter:
            to_compare.append(a)

        #print(to_compare)

        n_pair1 = 0

        while n_pair1 < len(to_compare)-1:
            n_pair2 = 0 #n_pair1 + 1
            while n_pair2 < len(to_compare):
                pair1 = to_compare[n_pair1]
                pair2 = to_compare[n_pair2]
                pairs.append({
                    'srcElem': annotations[n_pair1]['id'],
                    'trgElem': annotations[n_pair2]['id'],
                    'label': 'non-support',
                    'confidence': 1.
                })
                print("1: ", pair1)
                print("2: ", pair2)

                feat = generate_features(pair1, pair2)
                f.append(feat)

                n_pair2 += 1

            n_pair1 += 1

        counter += 1

    #print(f)

    return f, pairs


def create_csv(elements):
    global_id = 0
    final_annot = []

    for elem in elements:
        final_annot.append({
            'globalID': global_id,
            'text': elem['text'],
            'label': elem['label'],
            'id': elem['id']
        })

        global_id += 1

    return final_annot


def generate_features(pair1, pair2):
    features = {}
    # features["relationship"] = relationship.get_relationship(pair1, pair2) # 'support' 'non-support' categorical
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


# encode_features(elements)
#predict(elements)