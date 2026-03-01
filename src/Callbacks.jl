module Callbacks

using Evolutionary
using Printf
using ComponentArrays
using Dates
using ..Evaluation: accuracy
using ..Checkpoints: save_checkpoint

export CheckpointCallback, EvolutionaryObjective

function Evolutionary.trace!(
        record::Dict{String, Any}, objfun, state, population, method::ES, options
    )
    best_idx = argmin(state.fitness)
    record["σ"] = copy(state.strategies[best_idx].σ)
    record["θ"] = copy(state.fittest)
    record["L"] = state.fitness[best_idx]
    return
end

struct EvolutionaryObjective{M, S, XType, YType, L} <: Function
    model::M
    st::S
    axes::Any
    X::XType
    y::YType
    lossfn::L
end

function (obj::EvolutionaryObjective)(x)
    θ_current = ComponentArray(x, obj.axes)
    ŷ, _ = obj.model(obj.X, θ_current, obj.st)
    return obj.lossfn(ŷ, obj.y)
end

mutable struct CheckpointCallback{M, S, DL} <: Function
    i₀::Int
    block::Int
    checkpoint_Δi::Int
    i_timer::Float64
    complete_trace::Vector{Dict{String, Any}}
    model::M
    st::S
    axes::Any
    test_dataloader::DL
    best_test_acc::Float64
    prev_checkpoint::Ref{String}
    save_dir::String
    current_global_i::Int
end

function (cb::CheckpointCallback)(trace_record)
    if trace_record.iteration == 0
        return false
    end

    global_i = cb.i₀ + (cb.block - 1) * cb.checkpoint_Δi + trace_record.iteration
    cb.current_global_i = global_i

    current_time = time()
    elapsed = current_time - cb.i_timer
    cb.i_timer = current_time

    trace_dict = trace_record.metadata

    @printf(
        "Iter: %d \t Loss: %.6f \t Time: %.2fs\n",
        global_i, trace_dict["L"], elapsed
    )
    push!(cb.complete_trace, copy(trace_dict))

    if trace_record.iteration == cb.checkpoint_Δi
        θ_current = ComponentArray(trace_dict["θ"], cb.axes)

        recent_traces = @view cb.complete_trace[max(1, end - cb.checkpoint_Δi + 1):end]
        rolling_loss = sum(t["L"] for t in recent_traces) / length(recent_traces)

        test_acc = accuracy(cb.model, θ_current, cb.st, cb.test_dataloader)

        if test_acc > cb.best_test_acc
            cb.best_test_acc = test_acc

            checkpoint_data = Dict(
                "i" => global_i,
                "L" => trace_dict["L"],
                "θ" => trace_dict["θ"],
                "σ" => trace_dict["σ"],
                "rolling_loss" => rolling_loss,
                "test_acc" => test_acc,
                "complete_trace" => cb.complete_trace
            )

            save_checkpoint(
                checkpoint_data, test_acc, global_i, cb.prev_checkpoint, cb.save_dir
            )

            @printf(
                "\n[Checkpoint Saved] i: %d \t Current Loss: %.4f \t Rolling Loss: %.4f \t Test Acc: %.2f%%\n\n",
                global_i, trace_dict["L"], rolling_loss, test_acc
            )
        else
            @printf(
                "\n[Skipped Checkpoint] i: %d \t Current Loss: %.4f \t Rolling Loss: %.4f \t Test Acc: %.2f%% (Best: %.2f%%)\n\n",
                global_i, trace_dict["L"], rolling_loss, test_acc, cb.best_test_acc
            )
        end
    end

    return false
end

end
