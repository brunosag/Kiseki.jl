module Training

using Lux
using Random
using Printf
using Evolutionary
using ComponentArrays
using ..Data: load_MNIST
using ..Evaluation: accuracy
using ..Checkpoints: load_checkpoint
using ..Callbacks: CheckpointCallback, EvolutionaryObjective

export train_evolution, ESConfig

Base.@kwdef struct ESConfig
    μ::Int = 80
    λ::Int = 400
end

function train_evolution(
        model;
        I = 12000,
        batchsize = 1024,
        checkpoint_Δi = 100,
        resume_file = nothing,
        lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(),
        save_dir = pwd(),
        rng = Random.default_rng(),
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)

    θ_flat = ComponentArray(θ)
    N = length(θ_flat)
    axes_flat = getaxes(θ_flat)

    complete_trace = Dict{String, Any}[]
    best_test_acc = 0.0
    i₀ = 0

    checkpoint_data = isnothing(resume_file) ? nothing : load_checkpoint(resume_file)

    if !isnothing(checkpoint_data)
        θ_latest = checkpoint_data["θ"]
        σ_latest = checkpoint_data["σ"]
        i₀ = checkpoint_data["i"]
        best_test_acc = get(checkpoint_data, "test_acc", 0.0)
        if haskey(checkpoint_data, "complete_trace")
            complete_trace = checkpoint_data["complete_trace"]
        end
    else
        θ_latest = Vector(θ_flat)
        σ_latest = fill(0.01f0, N)
    end

    τ = Float32(1.0 / sqrt(2.0 * sqrt(N)))
    τ′ = Float32(1.0 / sqrt(2.0 * N))

    train_i = iterate(train_dataloader)
    total_blocks = (I - i₀) ÷ checkpoint_Δi
    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)

    cb_state = CheckpointCallback(
        i₀, 0, checkpoint_Δi, time(), complete_trace,
        model, s, axes_flat, test_dataloader,
        best_test_acc, prev_checkpoint, save_dir, i₀
    )

    for block in 1:total_blocks
        cb_state.block = block
        if train_i === nothing
            train_i = iterate(train_dataloader)
        end
        (X, y), dataloader_state = train_i
        train_i = iterate(train_dataloader, dataloader_state)

        objective = EvolutionaryObjective(model, s, axes_flat, X, y, lossfn)

        options = Evolutionary.Options(
            iterations = checkpoint_Δi,
            parallelization = :thread,
            rng = rng,
            abstol = -1.0,
            reltol = -1.0,
            successive_f_tol = 0,
            callback = cb_state,
        )

        poplt = [θ_latest .+ σ_latest .* randn(rng, Float32, N) for _ in 1:es_config.μ]

        algo = ES(
            initStrategy = AnisotropicStrategy(σ_latest, τ, τ′),
            recombination = Evolutionary.average,
            srecombination = Evolutionary.average,
            mutation = Evolutionary.gaussian,
            smutation = Evolutionary.gaussian,
            μ = es_config.μ,
            λ = es_config.λ,
        )

        Evolutionary.optimize(objective, Evolutionary.NoConstraints(), algo, poplt, options)

        last_trace = cb_state.complete_trace[end]
        θ_latest = last_trace["θ"]
        σ_latest = last_trace["σ"]
    end

    return cb_state.complete_trace
end

end
