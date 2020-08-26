import spacy
import collections

nlp = spacy.load('de_core_news_sm')


def get_sentences_from_ann(i):

    f_ann = open("../Corpus/%s.ann" % i, "r")
    line = f_ann.readline()
    annotations = []
    several_sentences_in_one_annot = []
    while line:
        l = line.split("\t")
        if l[0].startswith("T"):
            annotation = l[2]
            bla = [sent.string.strip() for sent in nlp(annotation).sents]
            if len(bla) > 1:
                several_sentences_in_one_annot.append(bla)
            annotations.append(annotation.strip())
        line = f_ann.readline()

    #for s in several_sentences_in_one_annot:
        #print("***", s)

    return annotations


def get_sentences_from_txt(i):

    f_in = open("../Corpus/%s.txt" % i, "r")
    text = f_in.read()
    sentences = [sent.string.strip() for sent in nlp(text).sents]
    return sentences


def compare_sentences():
    found = []
    not_found = []
    duplicates = []
    for i in range(0, 1000):
        sentences_txt = get_sentences_from_txt(i)
        sentences_ann = get_sentences_from_ann(i)

        #for a in sentences_ann:
            #print(a)

        #for t in sentences_txt:
            #print(t)
        found_sentences = []
        count = 0
        for a in sentences_ann:
            found_a = False

            for t in sentences_txt:

                if a in t:

                    found_sentences.append(t)
                    count += 1
                    found.append(a)
                    found_a = True

                    break

            if not found_a:
                not_found.append(a)
        #print(found_sentences)
        dup = [item for item, count in collections.Counter(found_sentences).items() if count > 1]
        duplicates.append(dup)
        #print("duplicates: ", [item for item, count in collections.Counter(found_sentences).items() if count > 1])

    print("not found: ", len(not_found), not_found)
    print("found: ", len(found))

    c=0
    for d in duplicates:
        print(c, d)
        c+=1

compare_sentences()
