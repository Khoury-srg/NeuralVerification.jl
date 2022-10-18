module NeuralVerification

using JuMP

#using GLPK, SCS, CPLEX # SCS only needed for Certify
# cheng-hack:
using GLPK, SCS # SCS only needed for Certify
using PicoSAT # needed for Planet
using LazySets, LazySets.Approximations
using Polyhedra, CDDLib

using LinearAlgebra
using Parameters
using Interpolations # only for PiecewiseLinear

import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

using Requires

# Runtime verification dependencies
using OrderedCollections
using ProgressMeter
using SparseArrays
using MosekTools
using Combinatorics

abstract type Solver end

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
JuMP.Model(solver::Solver) = Model(solver.optimizer)
# define a `value` function that recurses so that value(vector) and
# value(VecOfVec) works cleanly. This is only so the code looks nice.
value(var::JuMP.AbstractJuMPScalar) = JuMP.value(var)
value(vars::Vector) = value.(vars)
value(val) = val


include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")
include("utils/tree.jl")

function __init__()
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("utils/flux.jl")
end

export
    Solver,
    Network,
    AbstractActivation,
    PolytopeComplement,
    complement,
    # NOTE: not sure if exporting these is a good idea as far as namespace conflicts go:
    # ReLU,
    # Max,
    # Id,
    GeneralAct,
    PiecewiseLinear,
    Problem,
    TrackingProblem,
    TrainingProblem,
    AdaptingProblem,
    DomainShiftingProblem,
    DemandShiftingProblem,
    Result,
    BasicResult,
    CounterExampleResult,
    AdversarialResult,
    TrackingResult,
    ReachabilityResult,
    read_nnet,
    write_nnet,
    compute_output,
    solve,
    forward_network,
    check_inclusion,
    find_lipschitz,
    set_distance,
    get_gradient_bounds,
    get_gradient

solve(m::Model; kwargs...) = JuMP.solve(m; kwargs...)
export solve

# TODO: consider creating sub-modules for each of these.
include("optimization/utils/constraints.jl")
include("optimization/utils/objectives.jl")
include("optimization/utils/variables.jl")
include("optimization/nsVerify.jl")
include("optimization/convDual.jl")
include("optimization/duality.jl")
include("optimization/certify.jl")
include("optimization/iLP.jl")
include("optimization/mipVerify.jl")
include("optimization/nnDyn.jl")
export NSVerify, ConvDual, Duality, Certify, ILP, MIPVerify, NNDynTrack

include("reachability/utils/reachability.jl")
include("reachability/exactReach.jl")
include("reachability/maxSens.jl")
include("reachability/ai2.jl")
export ExactReach, MaxSens, Ai2, Ai2h, Ai2z, Box

include("satisfiability/bab.jl")
include("satisfiability/sherlock.jl")
include("satisfiability/reluplex.jl")
export BaB, Sherlock, Reluplex

include("satisfiability/planet.jl")
export Planet
include("adversarial/neurify.jl")
include("adversarial/reluVal.jl")
include("adversarial/adaptNeurify.jl")
include("adversarial/fastLin.jl")
include("adversarial/fastLip.jl")
include("adversarial/dlv.jl")
export ReluVal, Neurify, AdaptNeurify, FastLin, FastLip, DLV

include("runtime/intervalNet.jl")
include("runtime/branch_management.jl")
include("runtime/find_lipschitz.jl")
include("runtime/domainShiftingVerification.jl")
include("runtime/trainingVerification.jl")
include("runtime/adaptingVerification.jl")
include("runtime/demandShiftingVerification.jl")
export find_lipschitz, IntervalNet

const TOL = Ref(sqrt(eps()))
set_tolerance(x::Real) = (TOL[] = x)
export set_tolerance

end
