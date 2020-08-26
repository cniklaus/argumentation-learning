from features_relations import structural_number_tokens_source


def get_difference_number_tokens(pair1, pair2):
    length_pair1 = structural_number_tokens_source.get_number_of_tokens(pair1['text'])
    length_pair2 = structural_number_tokens_source.get_number_of_tokens(pair2['text'])

    if length_pair1 > length_pair2:
        difference = length_pair1 - length_pair2
    else:
        difference = length_pair2 -length_pair1

    return difference


# p1 = {'text': "Das wetter ist schön."}
# p2 = {'text': "Das Wetter, ist, schlecht."}
# print(get_difference_number_tokens(p1, p2))