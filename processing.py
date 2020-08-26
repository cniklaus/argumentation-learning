#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import logging
import time
import datetime
from delta import html
import json
import os
import pickle
import requests

sys.path.append('../')

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

# import feature extractors
from Q_was_helpful import Q_was_helpful_text_feature_extractor
from Q_high_quality import Q_high_quality_text_feature_extractor
from Q_critical_aspects import Q_critical_aspects_text_feature_extractor
from Q_constructive_suggestions import Q_constructive_suggestions_text_feature_extractor
from Q_task_related import Q_task_related_text_feature_extractor
from Q_highlights_weaknesses_strengths import Q_highlights_weaknesses_strengths_text_feature_extractor
from Q_easy_understand import Q_easy_understand_text_feature_extractor


##############################
##### Method definition ######
##############################


def loadModels(path):
    qCases = [
        'Q_was_helpful',
        'Q_high_quality',
        'Q_critical_aspects',
        'Q_constructive_suggestions',
        'Q_task_related',
        'Q_highlights_weaknesses_strengths',
        'Q_easy_understand'
    ]
    modelId = 1
    fvecs = {}
    models = {}

    try:
        for qCase in qCases:
            fvecpath = os.path.join(path, '%s.%s.fvec' % (qCase, modelId))
            modelpath = os.path.join(path, '%s.%s.model' % (qCase, modelId))
            fvecs[qCase] = pickle.load(open(fvecpath, 'rb'))
            models[qCase] = pickle.load(open(modelpath, 'rb'))
    except:
        print("\nError on laoding models - process.py\n")
        e = sys.exc_info()
        print(e)
    return fvecs, models


def toFraction(numeric_value):
    result = numeric_value
    if result > 7:
        result = 7
    if result < 0:
        result = 0
    return str(int(round(100 * result / 7.0)))


output_eng = {"ops": [
    {"insert": "Below you can find the results of the automated evaluation of your review. "
               "Please consider updating your text to improve your ratings.\n\n"
               "Quality of the feedback: \t\t\t\t\t\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nHelpfulness of the review:\t\t\t\t\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nReview identifies critical aspects:\t\t\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nReview provides constructive suggestions:\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    # {"insert":" out of 100\nFeedback is related to the task:  \t\t\t\t"},
    # {"attributes":{"bold":True},"insert":"57"},
    {"insert": " out of 100\nReview highlights weeknesses and strengths:\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    # {"insert":" out of 100\nFeedback is easy to understand:  \t\t\t\t"},
    # {"attributes":{"bold":True},"insert":"57"},
    {"insert": " out of 100"}
]}

output_de = {"ops": [
    {"insert": "Nachfolgend finden Sie eine automatisierte Bewertung Ihres Reviews. "
               "Nun können Sie Ihr Review überarbeiten um bessere Rating zu erhalten.\n\n"
               "Qualität des Reviews: \t\t\t\t\t\t\t\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nDas Review ist hilfreich:\t\t\t\t\t\t\t\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nDas Review identifiziert kritische aspekte:\t\t\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    {"insert": " out of 100\nDas Review beinhaltet konstruktive Vorschläge:\t\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    # {"insert":" out of 100\nFeedback is related to the task:  \t\t"},
    # {"attributes":{"bold":True},"insert":"57"},
    {"insert": " out of 100\nDas Review hebt schwächen sowie stärken hervor:\t"},
    {"attributes": {"bold": True}, "insert": "0"},
    # {"insert":" out of 100\nFeedback is easy to understand:  \t\t"},
    # {"attributes":{"bold":True},"insert":"57"},
    {"insert": " out of 100"}
]}


######################################
##### Method definition ##############
######################################


def apply_model(feedback_features, fvec, model):
    try:
        x_vec = fvec.transform(feedback_features)
        prediction = model.predict(x_vec)[0]
    except:
        print("\nError in apply model\n")
        e = sys.exc_info()[0]
        print(e)
    return prediction


def processFeedback(feedback_delta, fvecs, models):
    feedback_html = html.render(json.loads(feedback_delta)['ops'])

    # QUALITY
    try:
        qCase = "Q_high_quality"
        feedback_features = Q_high_quality_text_feature_extractor(feedback_html)
        quality_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
        output_eng['ops'][1]['insert'] = toFraction(quality_pred)
        output_de['ops'][1]['insert'] = toFraction(quality_pred)
        print("\n Q_high_quality: ", quality_pred, toFraction(quality_pred))
    except:
        print("\nError in calculating Quality\n")
        e = sys.exc_info()
        print(e)

    # HELPFULNESS
    try:
        qCase = "Q_was_helpful"
        feedback_features = Q_was_helpful_text_feature_extractor(feedback_html)
        helpfulness_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
        output_eng['ops'][3]['insert'] = toFraction(helpfulness_pred)
        output_de['ops'][3]['insert'] = toFraction(helpfulness_pred)
        print("\n Q_was_helpful: ", helpfulness_pred, toFraction(helpfulness_pred))
    except:
        print("\nError in calculating helpfulness\n")
        e = sys.exc_info()
        print(e)

    # CRITICAL ASPECTS
    try:
        qCase = "Q_critical_aspects"
        feedback_features = Q_critical_aspects_text_feature_extractor(feedback_html)
        critical_aspects_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
        output_eng['ops'][5]['insert'] = toFraction(critical_aspects_pred)
        output_de['ops'][5]['insert'] = toFraction(critical_aspects_pred)
        print("\n Q_critical_aspects: ", critical_aspects_pred, toFraction(critical_aspects_pred))
    except:
        print("\nError in calculating critical aspects\n")
        e = sys.exc_info()
        print(e)

    # SUGGESTIONS
    try:
        qCase = "Q_constructive_suggestions"
        feedback_features = Q_constructive_suggestions_text_feature_extractor(feedback_html)
        suggestions_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
        output_eng['ops'][7]['insert'] = toFraction(suggestions_pred)
        output_de['ops'][7]['insert'] = toFraction(suggestions_pred)
        print("\n Q_constructive_suggestions: ", suggestions_pred, toFraction(suggestions_pred))
    except:
        print("\nError in calculating suggestions\n")
        e = sys.exc_info()
        print(e)

    # TASK_RELATED
    # try:
    #    qCase = "Q_task_related"
    #    feedback_features = Q_task_related_text_feature_extractor(feedback_html)
    #    task_related_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
    #    output_eng['ops'][9]['insert'] = toFraction(task_related_pred)
    #    output_de['ops'][9]['insert'] = toFraction(task_related_pred)
    # except:
    #    print("\nError in calculating related\n")
    #    e = sys.exc_info()
    #    print(e)

    # WEAKNESSES AND STRENGTHS
    try:
        qCase = "Q_highlights_weaknesses_strengths"
        feedback_features = Q_highlights_weaknesses_strengths_text_feature_extractor(feedback_html)
        weaknesses_strengths_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
        output_eng['ops'][9]['insert'] = toFraction(weaknesses_strengths_pred)
        output_de['ops'][9]['insert'] = toFraction(weaknesses_strengths_pred)
        print("\nWeaknesses: ", weaknesses_strengths_pred, toFraction(weaknesses_strengths_pred))
    except:
        print("\nError in calculation strength and weaknesses\n")
        e = sys.exc_info()
        print(e)


    # ARGUMENTATION
    try:
        r = requests.post("http://localhost:5130/", json={"text":feedback_delta['reviewText']})
        output_eng['argmine'] = r.json()
        output_de['argmine'] = r.json()
    except:
        print("\nError in argumentation feedback\n")
        e = sys.exc_info()
        print(e)


    # EASY TO UNDERSTAND
    # try:
    #    qCase = "Q_easy_understand"
    #    feedback_features = Q_easy_understand_text_feature_extractor(feedback_html)
    #    easy_to_understand_pred = apply_model(feedback_features, fvecs[qCase], models[qCase])
    #    output_eng['ops'][13]['insert'] = toFraction(easy_to_understand_pred)
    #    output_de['ops'][13]['insert'] = toFraction(easy_to_understand_pred)
    #    print(easy_to_understand_pred, toFraction(easy_to_understand_pred))
    # except:
    #    print("\nError in calculation easy to understand\n")
    #    e = sys.exc_info()
    #    print(e)

    return {'output_eng': json.dumps(output_eng),
            'output_de': json.dumps(output_de)}





