"""
    Neurify(max_iter::Int64, tree_search::Symbol)

Neurify combines symbolic reachability analysis with constraint refinement to minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: AbstractPolytope
3. Output: AbstractPolytope

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Symbolic reachability analysis and iterative interval refinement (search).
- `max_iter` default `10`.

# Property
Sound but not complete.

# Reference
[S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana,
"Efficient Formal Safety Analysis of Neural Networks,"
*CoRR*, vol. abs/1809.08098, 2018. arXiv: 1809.08098.](https://arxiv.org/pdf/1809.08098.pdf)
[https://github.com/tcwangshiqi-columbia/Neurify](https://github.com/tcwangshiqi-columbia/Neurify)
"""

@with_kw struct Neurify
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.

    # Becuase of over-approximation, a split may not bisect the input set. 
    # Therefore, the gradient remains unchanged (since input didn't change).
    # And this node will be chosen to split forever.
    # To prevent this, we split each node only once if the gradient of this node doesn't change. 
    # Each element in splits is a tuple (gradient_of_the_node, layer_index, node_index).

    splits = Set() # To prevent infinity loop.

    # But in some cases (which I don't have an example, just a sense), 
    # it can happen that a node indeed needs to be split twice with the same gradient.
end

struct SymbolicInterval{F<:AbstractPolytope}
    Low::Matrix{Float64}
    Up::Matrix{Float64}
    interval::F
end

SymbolicInterval(x::Matrix{Float64}, y::Matrix{Float64}) = SymbolicInterval(x, y, Hyperrectangle([0],[0]))

# Data to be passed during forward_layer
struct SymbolicIntervalGradient
    sym::SymbolicInterval
    LΛ::Vector{Vector{Float64}} # mask for computing gradient.
    UΛ::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}} # range of a input interval (upper_bound - lower_bound)
end

function solve(solver::Neurify, problem::Problem)

    while !isempty(solver.splits) pop!(solver.splits) end
    problem = Problem(problem.network, convert(HPolytope, problem.input), convert(HPolytope, problem.output))

    reach_lc = problem.input.constraints
    output_lc = problem.output.constraints

    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m], base_name="x")
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)

    reach = forward_network(solver, problem.network, problem.input)
    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network) # This calls the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    result.status == :unknown || return result

    reach_list = [(reach, max_violation_con)]

    for i in 2:solver.max_iter
        length(reach_list) > 0 || return BasicResult(:holds)
        reach, max_violation_con = pick_out!(reach_list, solver.tree_search)
        intervals = constraint_refinement(solver, problem.network, reach, max_violation_con)
        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
            result.status == :violated && return result
            result.status == :holds || (push!(reach_list, (reach, max_violation_con)))
        end
    end
    return BasicResult(:unknown)
end

function check_inclusion(solver, reach::SymbolicInterval{HPolytope{N}}, output::AbstractPolytope, nnet::Network) where N
    # The output constraint is in the form A*x < b
    # We try to maximize output constraint to find a violated case, or to verify the inclusion.
    # Suppose the output is [1, 0, -1] * x < 2, Then we are maximizing reach.Up[1] * 1 + reach.Low[3] * (-1) 
    
    # return a result and the most violated constraint.
    reach_lc = reach.interval.constraints
    output_lc = output.constraints
    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model =Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m])
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)
    max_violation = -1e9
    max_violation_con = nothing
    for i in 1:size(output_lc, 1)
        obj = zeros(size(reach.Low, 2))
        for j in 1:size(reach.Low, 1)
            if output_lc[i].a[j] > 0
                obj += output_lc[i].a[j] * reach.Up[j,:]
            else
                obj += output_lc[i].a[j] * reach.Low[j,:]
            end
        end
        obj = transpose(obj)
        @objective(model, Max, obj * [x; [1]])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            y = compute_output(nnet, value(x))
            if !∈(y, output)
                return CounterExampleResult(:violated, value(x)), nothing
            end
            if objective_value(model) > output_lc[i].b
                if objective_value(model) - output_lc[i].b > max_violation
                    max_violation = objective_value(model) - output_lc[i].b
                    max_violation_con = output_lc[i].a
                end
            end
        else
            if ∈(value(x), reach.interval)
                println("Not OPTIMAL, but x in the input set")
                println("This is usually caused by open input set.")
                println("Please check your input constraints.")
                exit()
            end
            println("No solution, please check the problem definition.")
            exit()
        end
        
    end
    max_violation > 0 && return BasicResult(:unknown), max_violation_con
    return BasicResult(:holds), nothing
end

function constraint_refinement(solver::Neurify, nnet::Network, reach::SymbolicIntervalGradient, max_violation_con::AbstractVector{Float64})
    i, j, influence = get_nodewise_influence(solver, nnet, reach, max_violation_con)
    # We can generate three more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    nnet_new = Network(nnet.layers[1:i])
    reach_new = forward_network(solver, nnet_new, reach.sym.interval)
    C, d = tosimplehrep(reach.sym.interval)
    l_sym = reach_new.sym.Low[[j], 1:end-1]
    l_off = reach_new.sym.Low[[j], end]
    u_sym = reach_new.sym.Up[[j], 1:end-1]
    u_off = reach_new.sym.Up[[j], end]
    intervals = Vector{HPolytope{Float64}}(undef, 3)
    intervals[1] = HPolytope([C; l_sym; u_sym], [d; -l_off; -u_off])
    intervals[2] = HPolytope([C; l_sym; -u_sym], [d; -l_off; u_off])
    intervals[3] = HPolytope([C; -l_sym; -u_sym], [d; l_off; u_off])
    # intervals[4] = HPolytope([C; -l_sym; u_sym], [d; l_off; -u_off]) lower bound can not be greater than upper bound
    return intervals
end

function get_nodewise_influence(solver::Neurify, nnet::Network, reach::SymbolicIntervalGradient, max_violation_con::Vector{Float64})
    n_output = size(nnet.layers[end].weights, 1)
    n_length = length(nnet.layers)
    # We want to find the node with the largest influence
    # Influence is defined as gradient * interval width
    # The gradient is with respect to a loss defined by the most violated constraint.
    LG = transpose(copy(max_violation_con))
    UG = transpose(copy(max_violation_con))
    max_tuple = (0, 0, 0.0)
    for (k, layer) in enumerate(reverse(nnet.layers))
        i = n_length - k + 1
        if layer.activation != Id() 
            # Only split Relu nodes
            # A split over id node may not reduce over-approximation (the input set may not bisected).
            for j in 1:size(layer.bias,1)
                if (0 < reach.LΛ[i][j] < 1) && (0 < reach.UΛ[i][j] < 1)                    
                    max_gradient = max(abs(LG[j]), abs(UG[j]))
                    influence = max_gradient * reach.r[i][j]
                    if in((i,j, influence), solver.splits) # To prevent infinity loop
                        continue
                    end
                    # If we use > here, in the case that largest gradient is 0, this function will return (0, 0 ,0)
                    if influence >= max_tuple[3] 
                        max_tuple = (i, j, influence)
                    end
                end
            end
        end
        
        i >= 1 || break
        LG_hat = max.(LG, 0.0) * Diagonal(reach.LΛ[i]) + min.(LG, 0.0) * Diagonal(reach.UΛ[i])
        UG_hat = min.(UG, 0.0) * Diagonal(reach.LΛ[i]) + max.(UG, 0.0) * Diagonal(reach.UΛ[i])
        LG, UG = interval_map_right(layer.weights, LG_hat, UG_hat)
    end
    push!(solver.splits, max_tuple)
    return max_tuple
end

function forward_layer(solver::Neurify, layer::Layer, input)
    return forward_act(forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear for the first layer
function forward_linear(solver::Neurify, input::AbstractPolytope, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    r = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end

# Symbolic forward_linear
function forward_linear(solver::Neurify, input::SymbolicIntervalGradient, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ, input.r)
end

# Symbolic forward_act
function forward_act(input::SymbolicIntervalGradient, layer::Layer{ReLU})
    n_node, n_input = size(input.sym.Up)
    output_Low, output_Up = input.sym.Low[:, :], input.sym.Up[:, :]
    mask_lower, mask_upper = zeros(Float64, n_node), ones(Float64, n_node)
    interval_width = zeros(Float64, n_node)
    for i in 1:n_node
        println("node ",i)
        println(input)
        if upper_bound(input.sym.Up[i, :], input.sym.interval) <= 0.0
            # Update to zero
            mask_lower[i], mask_upper[i] = 0, 0
            output_Up[i, :] = zeros(n_input)
            output_Low[i, :] = zeros(n_input)
        elseif lower_bound(input.sym.Low[i, :], input.sym.interval) >= 0
            # Keep dependency
            mask_lower[i], mask_upper[i] = 1, 1
        else
            # Symbolic linear relaxation
            # This is different from ReluVal
            up_up = upper_bound(input.sym.Up[i, :], input.sym.interval)
            up_low = lower_bound(input.sym.Up[i, :], input.sym.interval)
            low_up = upper_bound(input.sym.Low[i, :], input.sym.interval)
            low_low = lower_bound(input.sym.Low[i, :], input.sym.interval)
            up_slop = up_up / (up_up - up_low)
            low_slop = low_up / (low_up - low_low)
            output_Low[i, :] =  low_slop * output_Low[i, :]
            output_Up[i, end] -= up_low
            output_Up[i, :] = up_slop * output_Up[i, :]
            mask_lower[i], mask_upper[i] = low_slop, up_slop
            interval_width[i] = up_up - low_low
        end
    end
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    LΛ = push!(input.LΛ, mask_lower)
    UΛ = push!(input.UΛ, mask_upper)
    r = push!(input.r, interval_width)
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end

function forward_act(input::SymbolicIntervalGradient, layer::Layer{Id})
    sym = input.sym
    n_node = size(input.sym.Up, 1)
    LΛ = push!(input.LΛ, ones(Float64, n_node))
    UΛ = push!(input.UΛ, ones(Float64, n_node))
    r = push!(input.r, ones(Float64, n_node))
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end


function upper_bound(map::Vector{Float64}, input::HPolytope)
    n = size(input.constraints, 1)
    m = size(input.constraints[1].a, 1)
    model =Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m])
    @constraint(model, [i in 1:n], input.constraints[i].a' * x <= input.constraints[i].b)
    @objective(model, Max, map' * [x; [1]])
    optimize!(model)
    return objective_value(model)
end


function lower_bound(map::Vector{Float64}, input::HPolytope)
    n = size(input.constraints, 1)
    m = size(input.constraints[1].a, 1)
    model =Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m])
    @constraint(model, [i in 1:n], input.constraints[i].a' * x <= input.constraints[i].b)
    @objective(model, Min, map' * [x; [1]])
    optimize!(model)
    return objective_value(model)
end
