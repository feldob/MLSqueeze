struct BoundaryExposer
    td::TrainingData
    sut::Function
    bs::BoundarySqueeze

    function BoundaryExposer(td::TrainingData, sut::Function, bs::BoundarySqueeze=BoundarySqueeze(td))
        # ensure compatible input type (all inputs) for BlackBoxOptim
        for ic in inputcols(td)
            td.df[:, ic] = convert(Vector{Float64}, td.df[:, ic])
        end

        return new(td, sut, bs)
    end
end

sut(be::BoundaryExposer) = be.sut
isminimal(be::BoundaryExposer, bc::BoundaryCandidate) = isminimal(be.bs, bc)
unique_outputs(be::BoundaryExposer) = unique_outputs(be.td)
tdframe(be::BoundaryExposer) = be.td.df
outputcol(be::BoundaryExposer) = outputcol(be.td)
inputcols(be::BoundaryExposer) = inputcols(be.td)

function getcandidate(df, inputs::Vector{Symbol})
    idx = rand(1:nrow(df))
    return collect(df[idx,inputs])
end

function apply_one_vs_all(be::BoundaryExposer; MaxTime::Int,
                                                iterations::Int,
                                                initial_candidates::Int)
    cands = BoundaryCandidate[]
    for uo in unique_outputs(be)
        df_uo = deepcopy(be.td.df)
        df_uo.tempoutput = map(o -> o == uo ? uo : "other", df_uo[:, outputcol(be)])
        td = TrainingData(sutname(be.td), df_uo; inputs = inputcols(be), output=:tempoutput)
        temp_be = BoundaryExposer(td, be.sut)
        newcands = apply(temp_be; MaxTime, iterations, initial_candidates)
        cands = vcat(cands, newcands)
    end

    return cands
end

function apply(be::BoundaryExposer; MaxTime=3::Int,
                                    iterations::Int=500,
                                    initial_candidates::Int=20,
                                    dist_output = isdifferent,
                                    one_vs_all::Bool=false,
                                    optimizefordiversity::Bool=true,
                                    add_new::Bool=true)
    if one_vs_all
        return apply_one_vs_all(be; MaxTime, iterations, initial_candidates)
    end

    cands = BoundaryCandidate[]
    removenext = 1
    doremovenext = true
    incumbent_diversity_diff = 0
    inputs = inputcols(be)

    gfs = groupby(tdframe(be), outputcol(be))
    for i in 1:iterations
        first, second = sample(1:length(gfs), 2; replace = false)
        init1 = getcandidate(gfs[first], inputs)
        init2 = getcandidate(gfs[second], inputs)
        cand = apply(be.bs, sut(be), init1, init2; MaxTime, dist_output)
        
        #TODO have a check for whether the search was successful... if points too far apart, not successf. also interesting - what are those cases? -> investigate.
        if !optimizefordiversity || length(cands) < initial_candidates + 1
            push!(cands, cand)
        else
            if !doremovenext
                push!(cands, cands[removenext])
                doremovenext = true
            end

            cands[removenext] = cand
            distances = pairwise(Euclidean(), left.(cands)) # TODO potential inconsistency cause of within-boundary order?
            neighbordistancesums = sum(sort(distances, dims=2)[:,2:3], dims=2)[:,1]

            diversity_diff = sum(neighbordistancesums) / length(cands)

            if incumbent_diversity_diff < diversity_diff
                incumbent_diversity_diff = diversity_diff
                if add_new
                    doremovenext = false
                end
            end

            removenext = argmin(neighbordistancesums) # remove closest to its two neighbor points (2d boundary has 2 neighbor points)
        end
        "************ $i" |> println
    end

    if optimizefordiversity && doremovenext
        cands = cands[setdiff(1:end, removenext)] # ensure that the very last "removenext" is respected
    end

    return cands
end

function swap_order!(r::DataFrameRow)
    r_copy = collect(r)
    clength = trunc(Int, length(r)/2)
    r[1:clength] = r_copy[(clength+1):end]
    r[(clength+1):end] = r_copy[1:clength]
end

function todataframe(candidates::AbstractVector{BoundaryCandidate}, sut::Function; output = :output)

    df_left = DataFrame()
    for (i, n) in enumerate(argnames(sut))
        df_left[!,n] = Vector{argtypes(sut)[i]}()
    end

    df_right = similar(df_left)

    for c in candidates
        push!(df_left, left(c))
        push!(df_right, right(c))
    end

    df_left[!, output] = map(r -> string(sut(r...)), eachrow(df_left))
    df_right[!, output] = map(r -> string(sut(r...)), eachrow(df_right))

    rename!(df_right, names(df_right) .=> "n_" .* names(df_right))
    foreach(n -> df_left[!, n] = df_right[!, n], names(df_right))

    n_output = Symbol("n_" * string(output))
    for r in eachrow(df_left)
       if r[n_output] < r[output]
            swap_order!(r)
       end
    end

    return df_left
end

function boundarylabel(df; output = :output)
    n_output = Symbol("n_" * string(output))
    return string(first(df[!, output])) * "-" * string(first(df[!, n_output]))
end

function plots(df::DataFrame, xind, yind, xlims, ylims; output = :output)
    nout = Symbol("n_" * string(output))
    gfs = groupby(df, [output, nout])
    xlabel, ylabel = names(df)[[xind, yind]]
    
    label = boundarylabel(gfs[1]; output)
    p = plot(gfs[1][!,xind],gfs[1][!,yind]; xlims, ylims,
                                            label, xlabel, ylabel,
                                            seriestype=:scatter)

    for i in 2:length(gfs)
        label = boundarylabel(gfs[i]; output)
        plot!(p, gfs[i][!,xind],gfs[i][!,yind]; xlims, ylims,
                                            label, xlabel, ylabel,
                                            seriestype=:scatter)
    end

    return p
end

function plots(df::DataFrame, limits; output=:output)
    ninputs = trunc(Int, (ncol(df)-2)/2)
    combs = collect(combinations(1:ninputs, 2))

    plts = Vector{Plots.Plot{Plots.GRBackend}}(undef, length(combs))
    for i in eachindex(combs)
        x = combs[i][1]
        y = combs[i][2]
        plts[i] = plots(df, x, y, limits[x], limits[y]; output)
    end
    
    return plts
end


function two_nearest_neighbor_distances(df::DataFrame, inputs::Vector{Symbol})
    d = pairwise(Euclidean(), Matrix(df[:, inputs])')
    return sort!(d, dims=2)[:, 2:3]
end

function two_nearest_neighbor_distances_stats(df::DataFrame, inputs::Vector{Symbol})
    d_nn = two_nearest_neighbor_distances(df, inputs)
    return mean(d_nn), std(d_nn)
end