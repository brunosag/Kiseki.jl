using Serialization
using Plots
using LaTeXStrings

if length(ARGS) != 1
    println(stderr, "Usage: julia visualize.jl <path_to_checkpoint.jls>")
    exit(1)
end

checkpoint_path = ARGS[1]

if !isfile(checkpoint_path)
    println(stderr, "Error: File not found at $checkpoint_path")
    exit(1)
end

data = deserialize(checkpoint_path)
trace = data["complete_trace"]

iterations = [t["i"] for t in trace]
loss = [t["L"] for t in trace]
σ_mean = [t["σ_mean"] for t in trace]
σ_var = [t["σ_var"] for t in trace]

σ_std = sqrt.(σ_var)

p1 = plot(
    iterations, loss,
    xlabel = "Iteration",
    ylabel = "Loss",
    title = "Objective Loss",
    color = :red,
    linewidth = 2
)

p2 = plot(
    iterations, sigma_mean,
    ribbon = σ_std,
    fillalpha = 0.3,
    label = L"\mathrm{E}[\sigma] ± \sqrt{\mathrm{Var}(\sigma)}",
    xlabel = "Iteration",
    ylabel = "σ",
    title = "Strategy Parameter Dynamics",
    color = :blue,
    linewidth = 2
)

fig = plot(p1, p2, layout = (2, 1), size = (800, 600), margin = 5Plots.mm)

display(fig)

println("Plot rendered. Press Enter to close the window and terminate the script.")
readline()
