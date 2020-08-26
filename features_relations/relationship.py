def get_relationship(pair1, pair2):
    relation = 0 # 'non-support'
    if pair1['supports'] == pair2['annotID']:
        relation = 1 #'support'

    return relation