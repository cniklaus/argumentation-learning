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


def create_csv(three_classes):
    f = []

    global_id = 0
    doc_id = 0

    final_annot = []
    keep = []

    for i in range(0,990):
        doc_annotations = []
        f_in = open("./Corpus/%s.txt" % i, "r")
        text = f_in.read()

        doc = nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]

        f_ann = open("./Corpus/%s.ann" % i, "r")
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


        last_claim = None
        for f in final_annot:
            f['heuristic_support'] = 'None'
            if f['label'] == 'Claim':
                last_claim = f
            if f['label'] in ("Claim", "Premise"):
                keep.append(f)
                if f['label'] == "Premise":
                    f['heuristic_support'] = last_claim['annotID']

        doc_id += 1

    with open('accuracy_annot.csv', mode='w') as csv_file:
        fieldnames = ['globalID', 'docID', 'annotID', 'text', 'label', 'supports', 'heuristic_support']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for f in keep: #final_annot
            writer.writerow(f)

    return keep


annots = create_csv(True)

count = 0.
correct = 0.
for a in annots:
    if a['supports'] == a['heuristic_support']:
        correct += 1
    count += 1

print("count: ", count)
print("correct: ", correct)

print("accuracy: ", correct/count)