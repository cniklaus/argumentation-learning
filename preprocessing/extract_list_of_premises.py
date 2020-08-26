import spacy

nlp = spacy.load('de_core_news_sm')


def get_premises_from_ann(i):

    f_ann = open("../Corpus/%s.ann" % i, "r")
    line = f_ann.readline()
    annotations = []
    while line:
        l = line.split("\t")
        if l[0].startswith("T"):
            annotation = l[2]

            label = l[1].split(" ")[0].strip()
            if label == "Premise":
                annotations.append(annotation.strip())
        line = f_ann.readline()
    return annotations


def create_list_of_premises(end):
    premises = []
    f_out = open("premises.txt", "w")
    for i in range(0,end):
        premises.append(get_premises_from_ann(i))
    flat_premises = [item for sublist in premises for item in sublist]
    print(flat_premises)

    for p in flat_premises:
        f_out.write(p+ "\n")



create_list_of_premises(100)