module Checkpoints

using Serialization
using Printf
using Dates

export load_checkpoint, save_checkpoint

function load_checkpoint(resume_file::String)
    if isfile(resume_file)
        @printf("Resuming from checkpoint: %s\n", resume_file)
        return deserialize(resume_file)
    end
    return nothing
end

function save_checkpoint(
        data::Dict,
        test_acc::Float64,
        global_iter::Int,
        prev_checkpoint::Ref{String},
        dir::String
    )
    time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")
    base_name = @sprintf("ES_A%04d_I%d_%s.jls", round(Int, test_acc * 100), global_iter, time_str)
    filename = joinpath(abspath(dir), base_name)

    serialize(filename, data)

    if !isempty(prev_checkpoint[]) && isfile(prev_checkpoint[])
        rm(prev_checkpoint[])
    end
    prev_checkpoint[] = filename

    return filename
end

end
