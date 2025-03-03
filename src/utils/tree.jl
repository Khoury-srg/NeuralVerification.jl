mutable struct Tree{T}
    data::Vector{T}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    leaves::OrderedSet{Int}
    size::Int
end
Tree(data) = Tree{typeof(data)}([data], [0], [Vector{Int}()], OrderedSet([1]), 1)

function add_child!(t::Tree, parent::Int, data)
    push!(t.data, data)
    push!(t.children, Vector{Int}())
    x = length(t.data)
    push!(t.leaves, x)
    push!(t.parent, parent)
    push!(t.children[parent], x)
    in(parent, t.leaves) && pop!(t.leaves, parent)
    t.size += 1
    return x
end

# connect two nodes in the tree. The parent of x doesn't change. Only the parent node will no longer be a leaf
function connect!(t::Tree, parent::Int, x::Int)
    push!(t.children[parent], x)
    in(parent, t.leaves) && pop!(t.leaves, parent)
    return x
end

function delete_node!(t::Tree, x::Int)
    t.size -= calc_subtree_size(t, x)
    if x in t.leaves
        pop!(t.leaves, x)
    end
    filter!(e->e≠x, t.children[t.parent[x]])
    length(t.children[t.parent[x]]) == 0 && push!(t.leaves, t.parent[x])
    t.parent[x] = 0
end

function delete_all_children!(t::Tree, x::Int)
    for c in t.children[x]
        delete_node!(t, c)
    end
end

function print_tree(t::Tree, x::Int = 1)
    for c in t.children[x]
        println(x, "->", c)
    end
    for c in t.children[x]
        print_tree(t, c)
    end
end

function print_tree_data(t::Tree, x::Int = 1)
    for c in t.children[x]
        println(t.data[x], "->", t.data[c])
    end
    for c in t.children[x]
        print_tree(t, c)
    end
end

function is_leaf(t::Tree, x::Int)
    return in(x, t.leaves)
end

function tree_size(t::Tree)
    return t.size
end

function calc_subtree_size(t::Tree, x::Int)
    sum = 0
    for c in t.children[x]
        sum += calc_subtree_size(t, c)
    end
    return sum+1
end

function test()
    t = Tree("a")
    add_child!(t, 1, "b")
    add_child!(t, 1, "c")
    add_child!(t, 2, "d")
    add_child!(t, 2, "e")
    print_tree(t)
    println("---")
    @assert tree_size(t) == 5
    @assert t.size == 5
    delete_node!(t, 2)
    @assert t.size == 2
    print_tree(t)
end
