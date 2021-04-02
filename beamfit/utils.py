import numpy as np


def get_image_and_weight(raw_images, dark_fields, mask):
    image = np.ma.masked_array(data=np.mean(raw_images, axis=0) - np.mean(dark_fields, axis=0), mask=mask)
    std_image = np.ma.masked_array(data=np.sqrt(np.std(raw_images, axis=0) ** 2 + np.std(dark_fields, axis=0) ** 2),
                                   mask=mask)
    image_weight = len(raw_images) / std_image ** 2
    return image, image_weight


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
