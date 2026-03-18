module optimizers

export AbstractOptimizer, AbstractOptimizerState
export LEEA, LEEAState
export SGD, SGDState
export init, step!

abstract type AbstractOptimizer end
abstract type AbstractOptimizerState end

function init end
function step! end

include("optimizers/leea.jl")
include("optimizers/sgd.jl")

using .leea: LEEA, LEEAState
using .sgd: SGD, SGDState

end
