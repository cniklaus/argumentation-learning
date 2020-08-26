def is_target_before_source(pair1, pair2):
    id1 = pair1['globalID']
    id2 = pair2['globalID']

    if id1 < id2:
        return True
    else:
        return False


# p1 = {'globalID': 7, 'text': "Das wetter ist schön."}
# p2 = {'globalID': 8, 'text': "Das Wetter ist schlecht."}
# print(is_target_before_source(p1, p2))