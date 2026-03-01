using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Printf, Evolutionary, ComponentArrays, Serialization, Dates
using MLDatasets: MNIST
using SimpleChains: SimpleChains

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

function accuracy(model, θ, s, dataloader)
    s_test = Lux.testmode(s)
    correct, total = 0, 0

    for (X, y) in dataloader
        pred = Array(first(model(X, θ, s_test)))
        correct += sum(onecold(pred) .== onecold(y))
        total += size(y, 2)
    end

    return (correct / total) * 100
end

function Evolutionary.trace!(record::Dict{String, Any}, objfun, state, population, method::ES, options)
    best_idx = argmin(state.fitness)
    record["σ"] = copy(state.strategies[best_idx].σ)
    record["θ"] = copy(state.fittest)
    record["L"] = state.fitness[best_idx]
    return
end

function train_evolution(
        model;
        rng = Random.default_rng(),
        batchsize = 2048,
        iterations = 12000,
        lossfn = CrossEntropyLoss(; logits = Val(true)),
        checkpoint_interval = 100,
        resume_file = nothing
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)

    θ_flat = ComponentArray(θ)
    N = length(θ_flat)

    complete_trace = Dict{String, Any}[]
    best_test_acc = Ref{Float64}(0.0)
    start_iter = 0

    if !isnothing(resume_file) && isfile(resume_file)
        checkpoint_data = deserialize(resume_file)
        θ_latest = checkpoint_data["θ"]
        σ_latest = checkpoint_data["σ"]
        start_iter = checkpoint_data["i"]
        best_test_acc[] = get(checkpoint_data, "test_acc", 0.0)

        if haskey(checkpoint_data, "complete_trace")
            complete_trace = checkpoint_data["complete_trace"]
        end

        @printf("Resuming from iteration %d...\n", start_iter)
    else
        θ_latest = Vector(θ_flat)
        σ_latest = fill(0.01f0, N)
    end

    μ_size = 80
    λ_size = 400
    τ = Float32(1.0 / sqrt(2.0 * sqrt(N)))
    τ′ = Float32(1.0 / sqrt(2.0 * N))

    train_iter = iterate(train_dataloader)
    total_blocks = (iterations - start_iter) ÷ checkpoint_interval
    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)
    current_global_iter = start_iter
    iteration_timer = Ref(time())
    axes = getaxes(θ_flat)

    for block in 1:total_blocks
        if train_iter === nothing
            train_iter = iterate(train_dataloader)
        end

        (X_static, y_static), dataloader_state = train_iter
        train_iter = iterate(train_dataloader, dataloader_state)

        let X = X_static, y = y_static, mdl = model, st = s, ax = axes, lf = lossfn
            function objective(x)
                θ_current = ComponentArray(x, ax)
                ŷ, _ = mdl(X, θ_current, st)
                return lf(ŷ, y)
            end

            function checkpoint_callback(trace_record)
                if trace_record.iteration == 0
                    return false
                end

                global_iter = start_iter + (block - 1) * checkpoint_interval + trace_record.iteration
                current_global_iter = global_iter

                current_time = time()
                elapsed = current_time - iteration_timer[]
                iteration_timer[] = current_time

                trace_dict = trace_record.metadata

                @printf("Iter %5d \t Loss: %.6f \t Time: %.4fs\n", global_iter, trace_dict["L"], elapsed)
                push!(complete_trace, copy(trace_dict))

                if trace_record.iteration == checkpoint_interval
                    θ_current = ComponentArray(trace_dict["θ"], ax)
                    train_acc = accuracy(mdl, θ_current, st, train_dataloader)
                    test_acc = accuracy(mdl, θ_current, st, test_dataloader)

                    if test_acc > best_test_acc[]
                        best_test_acc[] = test_acc

                        checkpoint_data = Dict(
                            "i" => global_iter,
                            "L" => trace_dict["L"],
                            "θ" => trace_dict["θ"],
                            "σ" => trace_dict["σ"],
                            "train_acc" => train_acc,
                            "test_acc" => test_acc,
                            "complete_trace" => complete_trace
                        )

                        time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")
                        base_name = @sprintf("ES_A%04d_I%d_%s.jls", round(Int, test_acc * 100), global_iter, time_str)
                        filename = joinpath(abspath(@__DIR__), base_name)

                        serialize(filename, checkpoint_data)

                        if !isempty(prev_checkpoint[]) && isfile(prev_checkpoint[])
                            rm(prev_checkpoint[])
                        end
                        prev_checkpoint[] = filename

                        @printf(
                            "\n[Checkpoint Saved] Iter %d \t Loss: %.4f \t Train Acc: %.2f%% \t Test Acc: %.2f%%\n\n",
                            global_iter, trace_dict["L"], train_acc, test_acc
                        )
                    else
                        @printf(
                            "\n[Skipped Checkpoint] Iter %d \t Loss: %.4f \t Train Acc: %.2f%% \t Test Acc: %.2f%% (Best: %.2f%%)\n\n",
                            global_iter, trace_dict["L"], train_acc, test_acc, best_test_acc[]
                        )
                    end
                end

                return false
            end

            options = Evolutionary.Options(
                iterations = checkpoint_interval,
                parallelization = :thread,
                rng = rng,
                abstol = -1.0,
                reltol = -1.0,
                successive_f_tol = 0,
                callback = checkpoint_callback,
            )

            poplt = [θ_latest .+ σ_latest .* randn(rng, Float32, N) for _ in 1:μ_size]

            algo = ES(
                initStrategy = AnisotropicStrategy(σ_latest, τ, τ′),
                recombination = Evolutionary.average,
                srecombination = Evolutionary.average,
                mutation = Evolutionary.gaussian,
                smutation = Evolutionary.gaussian,
                μ = μ_size,
                λ = λ_size,
            )

            Evolutionary.optimize(objective, Evolutionary.NoConstraints(), algo, poplt, options)
        end

        last_trace = complete_trace[end]
        θ_latest = last_trace["θ"]
        σ_latest = last_trace["σ"]
    end

    θ_best = ComponentArray(θ_latest, getaxes(θ_flat))
    train_acc = accuracy(model, θ_best, s, train_dataloader)
    test_acc = accuracy(model, θ_best, s, test_dataloader)

    return θ_best, complete_trace
end

model = ToSimpleChainsAdaptor((28, 28, 1))(
    Chain(
        Conv((3, 3), 1 => 8, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Dense(5 * 5 * 16 => 120, relu),
        Dense(120 => 10)
    )
)

rng = Xoshiro(42)
resume_target = isempty(ARGS) ? nothing : ARGS[1]
θ, result = train_evolution(model; rng = rng, resume_file = resume_target)
