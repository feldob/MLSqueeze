
isdifferent(a,b) = a == b ? 0 : 1

const FitnessBreakpoint = 10.0
function pd(i1, i2, o1, o2;
    dist_output = isdifferent, #TODO in future allow for ordinal distance or any other measure.
    dist_input = Euclidean())

    dout = dist_output(o1, o2)
    din = dist_input(i1, i2)

    if dout == 0.0
        return Inf # maximially penalize if same category - not our intention.
        #return FitnessBreakpoint + 1.0 / (din + Delta) # Push points away until we hopefully find a difference in the outputs
    else
        # addition of 1e-12 to ensure that we don't divide by zero (could happen in edge cases for non-deterministic suts)
        return FitnessBreakpoint - dout / (din + 1e-12) # When we have some output distance we start giving a benefit to points being close.
    end
end

# count of 1s must be exactly 1

function repair(x::Vector{Float64}, categorical_ranges::Vector{UnitRange{Int64}})
    if isempty(categorical_ranges)
        return x
    end

    x_copy = copy(x)
    for range in categorical_ranges
        x_copy[range] = zeros(length(range))
        x_copy[argmax(x[range])] = 1.0
    end
    return x_copy
end

function pd_fitness(_classifier::Function; dist_output = isdifferent,
                                            dist_input = Euclidean(),
                                            categorical_ranges = UnitRange{Int64}[])
    return (x::Vector{Float64}) -> begin
        _nargs = div(length(x),2)
        i1 = x[1:_nargs]
        i2 = x[_nargs+1:end]
        # i1_copy = repair(i1, categorical_ranges)
        # i2_copy = repair(i2, categorical_ranges)
        o1 = _classifier(i1...)
        o2 = _classifier(i2...)
        return pd(i1, i2, o1, o2; dist_output, dist_input)
    end
end

struct BoundaryCandidate 
    i_left
    i_right

    function BoundaryCandidate(bbo_sol::Vector)
        l = trunc(Int64, length(bbo_sol) / 2)
        return new(bbo_sol[1:l], bbo_sol[(l+1):end])
    end
end

left(bc::BoundaryCandidate) = bc.i_left
right(bc::BoundaryCandidate) = bc.i_right

struct BoundarySqueeze
    ranges
    dist_input
    dist_output
    delta
    npoints

    function BoundarySqueeze(ranges; Delta::Vector{Float64}=fill(1e-6, length(ranges)), npoints=20)
        return new(ranges, Euclidean(), isdifferent, Delta, npoints)
    end

    function BoundarySqueeze(td::TrainingData; npoints=20)
        ranges = deriveranges(td.df, inputcols(td))
        Delta = abs.(map(r -> r[2] - r[1], ranges)) ./ 1000 # create some reasonable small delta for acceptance depending on size of the range
        return new(ranges, Euclidean(), isdifferent, Delta, npoints)
    end
end

ranges(bs::BoundarySqueeze) = bs.ranges
distinput(bs::BoundarySqueeze) = bs.dist_input
distoutput(bs::BoundarySqueeze) = bs.dist_output
delta(bs::BoundarySqueeze) = bs.delta
npoints(bs::BoundarySqueeze) = bs.npoints

function isminimal(bs::BoundarySqueeze, c::BoundaryCandidate)
    indiv = Distances.colwise(Euclidean(), left(c), right(c)) .≤ delta(bs)
    return all(indiv)
end

function convergence_callback(bs::BoundarySqueeze)
    return (c::BlackBoxOptim.OptRunController) ->
    begin
        bc = BoundaryCandidate(best_candidate(c))
        if best_fitness(c) < Inf # valid pair (boundary has candidate on two sides of the boundary)
            
            if isminimal(bs, bc)
                    c.max_steps = BlackBoxOptim.num_steps(c)-1
                    println("--------")
                    println(c.max_steps)
                    println(best_fitness(c))
                    println(bc)
                    println("--------")
            end
        end
    end
end

function trivialpopulation(bs::BoundarySqueeze, cand1, cand2, categorical_ranges)
    seed = vcat(cand1, cand2)
    
    ncandinputs = length(cand1) * 2 # always same length

    _delta = vcat(delta(bs), delta(bs))

    # TODO must do the boxing - ensure in range
    pop = Matrix{Float64}(undef, ncandinputs, npoints(bs)-1)
    for i in 1:(npoints(bs)-1)
        pop[:, i] = seed .+ (rand(ncandinputs) .- 0.5) .* 2 .* _delta # introduce some random noise
    end

    # correct categorical entries (TODO no clue how exactly - want them to be close, but how does that represent current?)
    # cat_cols = vcat(categorical_ranges...)
    # cat_cols = vcat(cat_cols, length(cat_cols) .+ cat_cols)
    # for cat_col in cat_cols
    #     pop[cat_col,!] = .5
    # end

    return convert(Matrix{Float64}, [pop seed])
end

sumcolwise(m::Metric) = (a,b) -> sum(colwise(m, a, b))

function apply(bs::BoundarySqueeze, sut::Function, cand1, cand2;
                                                        MaxTime=3::Int,
                                                        dist_output = isdifferent,
                                                        dist_input = sumcolwise(Euclidean()),
                                                        categorical_ranges = UnitRange{Int64}[])
    Population = trivialpopulation(bs, cand1, cand2, categorical_ranges)

    # create ranges with margins
    _ranges = map(e -> (e[2][1]-delta(bs)[e[1]],e[2][2]+delta(bs)[e[1]]), enumerate(ranges(bs)))
    res = bboptimize(pd_fitness(sut; dist_output, dist_input, categorical_ranges);
                                SearchRange = vcat(_ranges, _ranges),
                                CallbackInterval = 0.0,
                                CallbackFunction = convergence_callback(bs),
                                MaxTime, Population)

    return BoundaryCandidate(best_candidate(res))
end