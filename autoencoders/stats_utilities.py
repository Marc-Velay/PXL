import numpy as np

def calc_inter_intra_variance(X_divided, num_per_class):
    per_class_average = []
    for classNum in range(0,10):
        per_class_average.append(np.average(np.array(X_divided[classNum]).T, axis=1))
    global_average = np.average(per_class_average, axis=0)

    inter_group = []
    for classNum in range(0,10):
        inter_group.append(np.sum(np.square(np.subtract(per_class_average[classNum],global_average))*num_per_class[classNum]))

    intra_group = []
    for classNum in range(0,10):
        intra_group.append(np.sum(np.sum(np.square(np.subtract(X_divided[classNum], per_class_average[classNum])), axis=0)))

    return np.sum(inter_group), np.sum(intra_group)
