@testset "test exposer" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES)
    be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))
    candidates = apply(be; iterations=10, initial_candidates=5)

    @test length(candidates) > 0
    for c in candidates
        @test isminimal(be, c)
    end

    df = todataframe(candidates, bmi_classification)
    plots(df, MLSqueeze.ranges(td))
end

@testset "one vs all exposer" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES)
    be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))
    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=true)
    df = todataframe(candidates, bmi_classification)

    # TODO add some real tests
    @test true
end

@testset "iris classifier test" begin
    df = CSV.read("../data/Iris.csv", DataFrame)
    
    inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
    output = :Species
    
    ranges = deriveranges(df, inputs)
    @test length(ranges) == 4
    @test ranges isa Vector{<:Tuple}

    td = TrainingData("iris", df; inputs, output)

    # create a model that can be used for classification
    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)
    be = BoundaryExposer(td, modelsut)
    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=false)
    df = todataframe(candidates, modelsut; output)

    plots(df, MLSqueeze.ranges(td); output)    
end

@testset "classifier test (ordinal titanic)" begin
    df = CSV.read("../data/titanic.csv", DataFrame)
    
    # TODO as far as missing handling not supported, filter out
    # ------------
    filter!(r -> !ismissing(r.Age), df)
    df.Age = convert(Vector{Float64}, df.Age)

    filter!(r -> !ismissing(r.Pclass), df)
    df.Pclass = convert(Vector{Float64}, df.Pclass)
    # ------------

    inputs = [:Pclass, :Age]
    output = :Survived
    
    ranges = deriveranges(df, inputs)

    td = TrainingData("titanic", df; inputs, output)

    # TODO do even for categorical, such as :Sex (setup another test)
    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)
    be = BoundaryExposer(td, modelsut)

    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=false)
    df = todataframe(candidates, modelsut; output)

    plots(df, MLSqueeze.ranges(td); output)
end

