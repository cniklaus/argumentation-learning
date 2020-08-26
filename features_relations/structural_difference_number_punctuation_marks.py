from features_relations import structural_number_punctuation_marks_source


def get_difference_number_of_punctuation_marks(pair1, pair2):
    number_punctuation_pair1 = structural_number_punctuation_marks_source.get_number_of_punctuation_marks(pair1['text'])
    number_punctuation_pair2 = structural_number_punctuation_marks_source.get_number_of_punctuation_marks(pair2['text'])

    if number_punctuation_pair1 > number_punctuation_pair2:
        difference = number_punctuation_pair1 - number_punctuation_pair2
    else:
        difference = number_punctuation_pair2 - number_punctuation_pair1

    return difference


# p1 = {'text': "Das wetter ist schön."}
# p2 = {'text': "Das Wetter, ist, schlecht."}
# print(get_difference_number_of_punctuation_marks(p1, p2))