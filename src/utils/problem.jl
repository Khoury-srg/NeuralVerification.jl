
"""
    PolytopeComplement

The complement to a given set. Note that in general, a `PolytopeComplement` is not necessarily a convex set.
Also note that `PolytopeComplement`s are open by definition.

### Examples
```julia
julia> H = Hyperrectangle([0,0], [1,1])
Hyperrectangle{Int64}([0, 0], [1, 1])

julia> PC = complement(H)
PolytopeComplement of:
  Hyperrectangle{Int64}([0, 0], [1, 1])

julia> center(H) ∈ PC
false

julia> high(H).+[1,1] ∈ PC
true
```
"""
struct PolytopeComplement{S<:LazySet}
    P::S
end

Base.show(io::IO, PC::PolytopeComplement) = (println(io, "PolytopeComplement of:"), println(io, "  ", PC.P))
LazySets.issubset(s, PC::PolytopeComplement) = LazySets.is_intersection_empty(s, PC.P)
LazySets.is_intersection_empty(s, PC::PolytopeComplement) = LazySets.issubset(s, PC.P)
LazySets.tohrep(PC::PolytopeComplement) = PolytopeComplement(convert(HPolytope, PC.P))
Base.in(pt, PC::PolytopeComplement) = pt ∉ PC.P
complement(PC::PolytopeComplement)  = PC.P
complement(P::LazySet) = PolytopeComplement(P)
Base.:(==)(pc1::PolytopeComplement, pc2::PolytopeComplement) = pc1.P == pc2.P
# etc.


"""
    Problem{P, Q}(network::Network, input::P, output::Q)

Problem definition for neural verification.

The verification problem consists of: for all  points in the input set,
the corresponding output of the network must belong to the output set.
"""
struct Problem{P, Q}
    network::Network
    input::P
    output::Q
end

struct TrackingProblem{P, Q, R, S}
    network::Network
    input::P
    output::Q
    output_ref::R
    output_cost::S
end

struct MPCProblem{P, Q, R, S}
    network::Network
    input::P
    output::Q
    output_ref::R
    output_cost::S
    horizon::Int
    dt::Float64
end


struct TrainingProblem{P, Q}
    networks::Array
    input::P
    output::Q
end

struct AdaptingProblem{P, Q}
    networks::Array
    input::P
    output::Q
end

struct DomainShiftingProblem{P, Q}
    network::Network
    inputs::Array{P}
    output::Q
end

struct DemandShiftingProblem{P, Q}
    network::Network
    input::P
    outputs::Array{Q}
end

"""
    Result
Supertype of all result types.

See also: [`BasicResult`](@ref), [`CounterExampleResult`](@ref), [`AdversarialResult`](@ref), [`ReachabilityResult`](@ref)
"""
abstract type Result end

status(result::Result) = result.status

function validate_status(st::Symbol)
    @assert st ∈ (:holds, :violated, :unknown) "unexpected status code: `:$st`.\nOnly (:holds, :violated, :unknown) are accepted"
    return st
end

"""
    BasicResult(status::Symbol)

Result type that captures whether the input-output constraint is satisfied.
Possible status values:\n
    :holds (io constraint is satisfied always)\n
    :violated (io constraint is violated)\n
    :unknown (could not be determined)
"""
struct BasicResult <: Result
    status::Symbol
end

"""
    CounterExampleResult(status, counter_example)

Like `BasicResult`, but also returns a `counter_example` if one is found (if status = :violated).
The `counter_example` is a point in the input set that, after the NN, lies outside the output set.
"""
struct CounterExampleResult <: Result
    status::Symbol
    counter_example::Vector{Float64}
    CounterExampleResult(s, ce) = new(validate_status(s), ce)
end

struct CounterExamplesResult <: Result
    status::Symbol
    counter_examples
    CounterExamplesResult(s, ce) = new(validate_status(s), ce)
end

"""
    AdversarialResult(status, max_disturbance)

Like `BasicResult`, but also returns the maximum allowable disturbance in the input (if status = :violated).
"""
struct AdversarialResult <: Result
	status::Symbol
	max_disturbance::Float64
    AdversarialResult(s, md) = new(validate_status(s), md)
end

"""
    ReachabilityResult(status, reachable)

Like `BasicResult`, but also returns the output reachable set given the input constraint (if status = :violated).
"""
struct ReachabilityResult <: Result
	status::Symbol
	reachable::Vector{<:AbstractPolytope}
    ReachabilityResult(s, r) = new(validate_status(s), r)
end


"""
    RuntimeResult(status, min_dis)

Like `BasicResult`, but also returns the maximum violation in the output constraints (min_dis > 0 iff status = :holds).
"""
struct RuntimeResult <: Result
	status::Symbol
	min_dis::Float64
    RuntimeResult(s, md) = new(validate_status(s), md)
end

"""
    TrackingResult(status, max_disturbance)

Like `BasicResult`, but also returns the maximum allowable disturbance in the input (if status = :violated).
"""
struct TrackingResult <: Result
	status::Symbol
    input::Vector{Float64}
    objective::Float64
    TrackingResult(s, i, o) = new(validate_status(s), i, o)
end

# Additional constructors:
TrackingResult(s) = TrackingResult(s, Float64[], 0)
CounterExampleResult(s) = CounterExampleResult(s, Float64[])
CounterExamplesResult(s) = CounterExamplesResult(s, [])
AdversarialResult(s)    = AdversarialResult(s, -1.0)
ReachabilityResult(s)   = ReachabilityResult(s, AbstractPolytope[])
