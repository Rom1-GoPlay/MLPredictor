import csv

def make_data(func, vals, distinction="0"):
    """
    create a csv where each row is a set of arguments (vals argument)
    for the given function (func argument) and the return value of this function.
    the "distincion" argument is just here to make multiple dataset of the same function
    without overwititting the previouse one.
    """

    cols = tuple(f"x{xn}" for xn in range(func.__code__.co_argcount)) + ("label",)

    # make csv with all that args as columns and output as label
    with open(f"data/generated/{func.__name__}_{distinction}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        csv_writer.writerow(cols)
        for val in vals:
            csv_writer.writerow(tuple(val) + (func(*val),))

def next_combine_len(*arrs):
    """return nbr of raws that make_inputs_from_arrs will return for the same arrs"""
    result = 1
    for arr in arrs:
        result *= len(arr)
    return result

def make_inputs_from_arrs(*arrs):
    """
    return a list of every combinations of values between given arrs
    """
    nbr_of_raws = next_combine_len(*arrs)
    result = [[] for _ in range(nbr_of_raws)]
    parcoured_raws = 1
    for arr in arrs:
        parcoured_raws *= len(arr)
        switch_each = nbr_of_raws // parcoured_raws
        val_idx = 0
        for raw_idx in range(nbr_of_raws):
            # calculate the index in arr of the value :
            val_idx += 1 if raw_idx % switch_each == 0 and raw_idx != 0 else 0
            val_idx = val_idx % len(arr)
            # add the value to the right list :
            result[raw_idx].append(arr[val_idx])

    return result


# Here are some functions you can use to create a dataset:

def linear(x):
    return 5 * x

def affine(x):
    return 5 * x + 10

def two_params(x0, x1):
    a = 6
    b = 8
    c = 2
    return  a * x0 + b * x1 + c

def three_params(x0, x1, x2):
    a = 6
    b = 8
    c = 2
    d = 80
    return  a * x0 + b * x1 + c * x2 + d


# here are some input arrays you can use to create a dataset:

input_array1 = ([i/10 for i in range(0, 100, 1)],)
input_array2 = ([i for i in range(100, 200, 20)], [i for i in range(10, 100, 4)],)
input_array3 = ([i for i in range(100, 200, 20)], [i for i in range(10, 100, 4)], [i for i in range(10, 100, 4)],)

# to create a data set:

make_data(three_params, make_inputs_from_arrs(*input_array3), 0)