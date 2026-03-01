module Data

using MLDatasets: MNIST
using MLUtils: DataLoader
using OneHotArrays: onehotbatch

export load_MNIST

function load_MNIST(rng; batchsize::Int)
    preprocess(data) = (
        reshape(data.features, 28, 28, 1, :),
        onehotbatch(data.targets, 0:9),
    )

    X_train, y_train = preprocess(MNIST(:train))
    X_test, y_test = preprocess(MNIST(:test))

    return (
        DataLoader((X_train, y_train); batchsize, shuffle = true, rng),
        DataLoader((X_test, y_test); batchsize, shuffle = false),
    )
end

end
