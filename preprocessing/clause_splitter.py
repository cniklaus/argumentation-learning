import spacy
from spacy.lang.de.examples import sentences
#from collections import OrderedDict
#import numpy as np

nlp = spacy.load('de_core_news_sm')

doc = nlp("Weil die Sonne scheint, ist es warm, nachdem ich ein Eis, das sehr lecker war, gegessen habe.")
print(doc.text)
#for token in doc:
   # print(token.text, token.pos_, token.dep_)

#TODO add recursion!
#TODO check for empty main clauses!


def split_relative_clauses(sentence):
    relc = []
    main = []
    rc_left = []
    rc_right = []
    start = 0
    for token in sentence:
        print(token, token.i, token.dep_)

        if token.dep_ == "rc":
            start = token.left_edge.i
            rel_clause = sentence[token.left_edge.i: token.right_edge.i+1]
            rc_right.append(token.i+1)
            rc_left.append(token.left_edge.i)
            relc.append(rel_clause)

    count = 0
    for j in rc_left:
        print(start, rc_left, rc_right)
        end = j
        if start == end:
            end = rc_left[count]
        main1 = sentence[start: rc_right[count]]
        start = rc_right[count]
        count += 1
        if len(main1) > 1:
            main.append(main1)
    print("main: ", main)
    print("relcl: ", relc)


def split_adverbial_clauses(sentence):
    advclauses = []
    main = []
    advcl_left = []
    advcl_right = []
    for token in sentence:
        if token.dep_ == "cp":
            adverbial_clause = sentence[token.left_edge.i : token.head.i+1]
            advcl_right.append(token.head.i+1)
            advcl_left.append(token.left_edge.i)
            advclauses.append(adverbial_clause)

    start = 0
    count = 0
    for j in advcl_left:
        end = j
        main1 = sentence[start: end]
        start = advcl_right[count]
        count += 1
        if len(main1) > 1:
            main.append(main1)
    print(main)
    print(advclauses)

    for a in advclauses:
        split_relative_clauses(a)


def split_coordinate_clauses1(sentence):
    for token in sentence:
        if token.dep_ == "oc":
            rel_clause = sentence[token.left_edge.i : token.head.i+1]
            main1 = sentence[:token.left_edge.i]
            main2 = sentence[token.head.i+1: ]
            print(rel_clause)
            print(main1)
            print(main2)


def split_coordinate_clauses2(sentence):
    for token in sentence:
        if token.dep_ == "cd":
            rel_clause = sentence[token.left_edge.i : token.head.i+1]
            main1 = sentence[:token.left_edge.i]
            main2 = sentence[token.i: ]
            print(rel_clause)
            print(main1)
            print(main2)


#def split_into_clauses(sentence):



#split_relative_clauses(doc)
split_adverbial_clauses(doc)
#split_coordinate_clauses1(doc)
#split_coordinate_clauses2(doc)