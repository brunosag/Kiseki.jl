module Evaluation

using Lux: testmode
using OneHotArrays: onecold

export accuracy

function accuracy(model, θ, s, dataloader)
    s_test = testmode(s)
    correct, total = 0, 0

    for (X, y) in dataloader
        pred = Array(first(model(X, θ, s_test)))
        correct += sum(onecold(pred) .== onecold(y))
        total += size(y, 2)
    end

    return (correct / total) * 100
end

end
