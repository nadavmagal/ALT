def tag_odd_even(labels):
    labels = labels % 2
    labels[labels == 0 ] = -1
    return labels


def tag_is_big_from_5(labels):
    labels[labels >= 5] = 1
    labels[labels < 5] = -1
    return labels


def tag_bd_date(labels):
    labels[(labels == 3) | (labels == 1) | (labels == 9) | (labels == 0)] = 1
    labels[labels != 1] = -1
    return labels
