#TODO can't handle missing data yet

function assertdataframe(df::DataFrame; inputs, output)
    @assert !isempty(df) "training data frame is empty"
    @assert all(map(i -> i ∈ names(df), string.(inputs))) "the inputs $inputs must all exist in $(names(df))"
    @assert ncol(df) ≥ 2 "there must be a class and at least one feature"
    return df
end

argnames(f::Function) = string.(Base.method_argnames(methods(f).ms[1])[2:end])
argtypes(f::Function) = methods(f).ms[1].sig.parameters[2:end]

function defaultranges(f::Function)
    types = argtypes(f)
    return [tuple.(typemin.(types), typemax.(types))...]
end

funcname(f) = methods(f).ms[1].name

function deriveranges(df::DataFrame, inputs::Vector{Symbol})
    ranges = Vector{Tuple{Float64,Float64}}(undef, length(inputs))
    for (i, input) in enumerate(inputs)
        ranges[i] = (minimum(Float64, df[!,input]), maximum(Float64, df[!,input]))
    end
    return ranges
end

defaultinputs(df::DataFrame) = Symbol.(names(df)[1:end-1])

#TODO in first iteration only numbers as inputs that allow random sampling.
struct TrainingData
    sutname::String
    df::DataFrame
    inputs::Vector{Symbol}
    output::Symbol
    ranges::Vector{Tuple{Float64,Float64}}

    function TrainingData(name, df::DataFrame;
                                    inputs=defaultinputs(df),
                                    output = Symbol(names(df)[end]))
        return new(name, assertdataframe(df; inputs, output), inputs, output, deriveranges(df, inputs))
    end

    # TODO add support for other distributions (even for numeric types)
    function TrainingData(sut::Function; ranges = defaultranges(sut),
                                            npoints = 1000,
                                            sampler = rand)
        _names = argnames(sut)
        types = argtypes(sut)
        vectors = map(i -> Vector{types[i]}(undef, npoints), eachindex(_names))

        # randomly sample values and get result
        for i in 1:npoints
            for arg in eachindex(_names)
                min, max = ranges[arg]
                type = types[arg]
                vectors[arg][i] = min + sampler(type) * abs(max - min)
            end
        end

        df = DataFrame(_names .=> vectors)
        df.output = map(r -> sut(r...), eachrow(df))

        return new(string(funcname(sut)), df, defaultinputs(df), :output, ranges)
    end
end

outputcol(td::TrainingData) = td.output
inputcols(td::TrainingData) = td.inputs
ranges(td::TrainingData) = td.ranges
outputtype(td::TrainingData) = eltype(td.df[!, outputcol(td)])
sutname(td::TrainingData) = td.sutname
npoints(td::TrainingData) = nrow(td.df)
classproblem(td::TrainingData) = (outputtype == Bool) || (!(outputtype(td) <: Real) && length(unique(td.df[!, outputcol(td)])) < 30)
ninputs(td::TrainingData) = length(inputcols(td))
tofile(td::TrainingData, filename="data/$(sutname(td)).csv"::String) = CSV.write(filename, td.df)
unique_outputs(td::TrainingData) = unique(td.df[!, outputcol(td)])

function extremes(col,slack=.1)
    min, max = minimum(col), maximum(col)
    return min - slack * abs(min), max + slack * abs(max)
end

function plots(td::TrainingData, indx1::Int, indx2::Int)
    xlims = extremes(td.df[!,indx1])
    ylims = extremes(td.df[!,indx2])
    
    gfs = groupby(td.df, td.output)
    label = gfs[1][1, td.output] |> string
    p = plot(gfs[1][!,indx1],gfs[1][!,indx2]; xlims, ylims, label,
                                            seriestype=:scatter)

    for i in 2:length(gfs)
        label = gfs[i][1, td.output]
        plot!(p, gfs[i][!,indx1],gfs[i][!,indx2]; xlims, ylims, label,
                                            seriestype=:scatter)
    end

    return p
end

function plots(td::TrainingData)
    @assert classproblem(td) "The problem is not a classification problem"

    combs = collect(combinations(1:ninputs(td), 2))
    plts = Vector{Plots.Plot{Plots.GRBackend}}(undef, length(combs))
    for i in eachindex(combs)
        plts[i] = plots(td, combs[i][1], combs[i][2])
    end
    
    return plts
end

function getmodelsut(td::TrainingData; model, fit)
    global temp_model = model
    training = hcat([td.df[!, i] for i in inputcols(td)]...) # OBS using all data for training
    fit(temp_model, training, td.df[!, outputcol(td)])

    # Create the parameter tuple expression (correct impl)
    #TODO might not always be Float64, can be other numeric types.
    #global temp_type = eltype(typeof(inputcols(td)[1]))
    param_tuple_expr = Expr(:tuple, [:( $(Symbol(p))::Float64 ) for p in inputcols(td)]...)

    # extract the expression
    exprstring = ":(DecisionTree.predict(temp_model, [" * join(string.(inputcols(td)), ", ") * "]))"
    function_body_expr = eval(Meta.parse(exprstring))

    anon_func_expr = Expr(:->, param_tuple_expr, function_body_expr)

    return eval(anon_func_expr)
end

# # TODO seemingly "the standard of doing ML with Julia", but veeeery slow.
# TODO no automatic extraction of the evaluation function with correct names ( requires manual setting so far).
# function getmodelsut(df, inputs, output)

#     X = df[!, inputs]

#     df[!, output] = categorical(df[!, output]) # assumption that outputs are categorical
#     y = df[!, output]

#     Tree = @load DecisionTreeClassifier pkg=DecisionTree
#     tree = Tree()
#     model = MLJ.fit!(machine(tree, X, y))

#     evfunc = (SepalLengthCm::Float64, SepalWidthCm::Float64, PetalLengthCm::Float64, PetalWidthCm::Float64) -> begin
#         Xnew = similar(X, 0)
#         push!(Xnew, (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm))
#         return MLJ.predict_mode(model, Xnew)[1] |> string
#     end

#     return evfunc
# end