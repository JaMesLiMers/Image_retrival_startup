
def P_N(label_result: list, N):
    """
    Calculate P of a result list.

    Args:
        result: (list) A list of result which 1 is positive and 0 is negative predict, 
            e.g.:
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        N: (int) An int which specify this algorithm get the top N result and calculate 
            the precition.
        
    Return:
        Precition value of the result.
    """
    assert N <= len(label_result), "N must smaller than result length!"
    result = sum(label_result[:N]) / N
    return result

def AP_N(label_result: list, N):
    """
    Calculate the MAP@N of a result list.

    Args:
        label_result: (list) A list of result which 1 is positive and 0 is negative predict, 
            e.g.:
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        N: (int) An int which specify this algorithm get the top N's MAP value.

    Return:
        MAP of N true value.
    """
    assert N <= len(label_result), "N must smaller than result length!"
    label_result = label_result[:N]

    # true label idx
    true_idx = [i for i in range(len(label_result)) if label_result[i] == 1]

    result = 0
    for i in true_idx:
        result += P_N(label_result, i +1)
        
    result = result / len(true_idx)
    return result
    

if __name__ == "__main__":
    # test 
    test_label_1 = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    test_result_1 = AP_N(test_label_1, 10)
    # 0.78
    print(test_result_1)
    test_label_2 = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    test_result_2 = AP_N(test_label_2, 10)
    # 0.52
    print(test_result_2)


