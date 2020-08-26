import spacy

nlp = spacy.load('de_core_news_sm')


def extract_sentences_txt(text):
    sentences = [sent.string.strip() for sent in nlp(text).sents]
    sentences_without_final_punct = []
    for sent in sentences:
        se = nlp(sent)
        if se[-1].is_punct:
            s = se[:-1]
        else:
            s = se
        sentences_without_final_punct.append(s.text)

    return sentences_without_final_punct


def create_data_set():

    annotation_is_full_sentence = 0
    annotation_is_partial_sentence = 0
    annots = []
    for i in range(0,1000):
        f_in = open("../Corpus/%s.txt" % i, "r")
        text = f_in.read()
        sentences = extract_sentences_txt(text)

        f_ann = open("../Corpus/%s.ann" % i, "r")
        line = f_ann.readline()
        annotations = []
        while line:
            l = line.split("\t")
            if l[0].startswith("T"):
                annotation = l[2]
                annotations.append(annotation.strip())
                annots.append(annotation.strip())
            line = f_ann.readline()

        for a in annotations:
            #print(a, sentences)
            if a in sentences:
                annotation_is_full_sentence += 1
            else:
                print(a, sentences, "\n")
                annotation_is_partial_sentence += 1

    print(len(annots))
    print(annotation_is_full_sentence)
    print(annotation_is_partial_sentence)


create_data_set()