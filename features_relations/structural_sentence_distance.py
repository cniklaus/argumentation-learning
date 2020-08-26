def get_sentence_distance(pair1, pair2):
    id1 = pair1['globalID']
    id2 = pair2['globalID']

    if id1 > id2:
        distance = id1 - id2
    else:
        distance = id2 - id1

    return distance


# p1 = {'globalID': 7, 'text': "Das wetter ist schön."}
# p2 = {'globalID': 5, 'text': "Das Wetter ist schlecht."}
# print(get_sentence_distance(p1, p2))