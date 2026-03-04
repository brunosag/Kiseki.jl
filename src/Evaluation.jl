module Evaluation

using Lux: testmode, gpu_device, cpu_device
using OneHotArrays: onecold

export accuracy


function accuracy(model, θ, s, dataloader)
    dev = gpu_device()
    cpu_dev = cpu_device()

    s_test = dev(testmode(s))
    θ_dev = dev(θ)

    correct, total = 0, 0

    for (X, y) in dataloader
        X_dev = dev(X)

        pred = first(model(X_dev, θ_dev, s_test))
        pred_cpu = cpu_dev(pred)

        correct += sum(onecold(pred_cpu) .== onecold(y))
        total += size(y, 2)
    end

    return (correct / total) * 100.0
end


end
