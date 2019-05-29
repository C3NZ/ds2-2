import numpy as np


def main() -> None:
    # Create a vector with np array
    vector = np.array([5, 2, 1])
    print(vector)
    print(vector.shape)

    # Create another vector (really a matrix)
    vector = np.array([[5, 2, 1], [4, 3, 2]])
    print(vector)
    print(vector.shape)

    # Flip the axes on our vector
    transposed_vector = np.transpose(vector)
    print(transposed_vector)

    # Create a matrix with np.array that is 2x3
    matrix = np.array([[1, 2, 3], [2, 0, 1]])
    print(matrix)
    print(matrix.shape)
    print(matrix.size)

    # Transpose our matrix to become a 3x2 matrix
    print("---Transposing matrix---")
    transposed_matrix = np.transpose(matrix)
    print(transposed_matrix)
    print(transposed_matrix.shape)
    print(transposed_matrix.size)

    #
    print("---Reshaping matrix---")
    print(matrix.reshape(3, 2))

    print("---reshaping matrix into 3 dimensions---")
    print(matrix.reshape(3, 1, 2))
    print(matrix.reshape(2, 3, 1))

    print("---Building a 3d tensor in numpy---")
    tensor_3d = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    )
    print(tensor_3d)

    print("---Reshaping 3d tensor to have 3 cubes, 2 planes, and 2 points")
    print(tensor_3d.reshape(3, 2, -1))
    print("---Reshpaing 3d tensor to be 4 planes with 3 points")
    print(tensor_3d.reshape(4, 3))
    print("---Reshpaing 3d tensor to be 3 planes with 4 points")
    print(tensor_3d.reshape(3, 4))
    print("---Reshaping 3d tensor to be 2 cubes with 3 planes with 2 points")
    print(tensor_3d.reshape(2, 3, 2))

    print("Create a 4d tensor ")
    tensor_4d = np.zeros((2, 3, 5, 5))


if __name__ == "__main__":
    main()
