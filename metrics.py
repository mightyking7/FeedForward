
def accuracy(actual, predicted):
    """
    Computes accuracy of classifications from the network
    :param actual: classes of MNIST images
    :param predicted: predicted classes from network
    :return: accuracy
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0