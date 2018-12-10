import random


def random_adjust_mask(mask, cutoff=0.5):
    """
    randomly set some False to be True; the mask is used to select two categories of rows in a data.frame;
    select all for one category but some of the other category
    :param mask: a pd.Series of boolean or a boolean column of a data.frame
    :param cutoff: a float
    :return:
    """
    mask = mask.copy()
    for i in mask[~mask].index:
        if random.uniform(0, 1) < cutoff:
            mask[i] = True
    return mask