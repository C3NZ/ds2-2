import numpy as np


def show_dot_prod() -> None:
    """
        Demonstrate the dot product of two matricies. One being the 
        transposed version of itself.
    """
    matrix = np.array([[4, 2, 3], [2, 0, 1]])
    transposed_matrix = np.transpose(matrix)
    dot_prod = np.dot(matrix, transposed_matrix)
    other_dot_prod = np.dot(transposed_matrix, matrix)
    print(dot_prod)
    print(other_dot_prod)


def element_wise_mult() -> None:
    """
        Demonstrate what element wise multiplcation would look like with
        matricies from numpy
    """
    first_mat = np.array([[1, 2], [3, 4]])
    second_mat = np.array([[5, 6], [7, 8]])

    # Show communitive properties of multiplying two matricies of the same dimension
    # by eachother. (using python and np.multiply())
    print(np.multiply(first_mat, second_mat))
    print(np.multiply(second_mat, first_mat))

    print(first_mat * second_mat)
    print(second_mat * first_mat)


def nn_with_numpy() -> None:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    output_neurons = 1
    hidden_layer_neurons = 3

    features = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
    desired_output = np.array([[1], [1], [0]])
    hidden_layer_weight = np.random.uniform(
        size=(features.shape[1], hidden_layer_neurons)
    )
    hidden_layer_bias = np.random.uniform(size=(1, hidden_layer_neurons))
    output_layer_weight = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
    output_layer_bias = np.random.uniform(size=(1, output_neurons))

    hidden_layer_input = np.dot(features, hidden_layer_weight) + hidden_layer_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = (
        np.dot(hidden_layer_output, output_layer_weight) + output_layer_bias
    )

    output = sigmoid(output_layer_input)
    print(output)


if __name__ == "__main__":
    show_dot_prod()
    element_wise_mult()
    nn_with_numpy()
