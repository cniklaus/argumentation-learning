import spacy
from preprocessing import read_input_text

nlp = spacy.load('de_core_news_sm')

#doc = nlp(sentences[1])
#print(doc.text)
#for token in doc:
    #print(token.text, token.pos_, token.dep_)

causal_markers = ["da", "denn", "weil"]
consecutive_markers = ["also", "dadurch", "daher", "darum", "deshalb", "deswegen", "drum"]
adversative_markers = ["anstatt", "anderenfalls", "andererseits", "einerseits", "hingegen", "jedoch", "sondern", "statt", "stattdessen", "vielmehr", "wiederum", "zum einen", "zum anderen"]
concessive_markers = ["allerdings", "dennoch", "gleichwohl", "nichtsdestotrotz", "nichtsdestoweniger", "obschon", "obwohl", "trotzdem", "wenngleich", "wobei", "zwar"]
conditional_markers = ["falls", "ob", "sofern", "wenn"]

argumentative_discourse_markers = causal_markers + consecutive_markers + adversative_markers + concessive_markers + conditional_markers


# def get_number_of_discourse_markers(text):
#    dep_list = [token.dep_ for token in text]
#    counter = 0
#    print(dep_list)
#    for d in dep_list:
#        if d == 'dm':
#            counter += 1
#    print(counter)
#    return False


def get_number_of_argumentative_discourse_markers(text):
    token_list = [token.text.lower() for token in text]

    counter = 0
    for t in token_list:
        if t in argumentative_discourse_markers:
            #print(t)
            counter += 1

    return counter


# print(get_number_of_argumentative_discourse_markers(read_input_text.read_input(11)))
