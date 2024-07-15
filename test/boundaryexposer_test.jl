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
    candidates = apply(be; iterations=10, initial_candidates=5, strategy=OneVsAll("Obese"))
    df = todataframe(candidates, bmi_classification)
end

@testset "all vs all exposer" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES)
    be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))
    candidates = apply(be; iterations=10, initial_candidates=5, strategy=AllVsAll())
    df = todataframe(candidates, bmi_classification)
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
    candidates = apply(be; iterations=10, initial_candidates=5)
    df = todataframe(candidates, modelsut; output)

    plots(df, MLSqueeze.ranges(td); output)
end

@testset "classifier test (ordinal titanic)" begin
    df = CSV.read("../data/titanic.csv", DataFrame)
    
    # missing must be removed or handled otherwise
    # ------------
    dropmissing!(df)
    df.Age = convert(Vector{Float64}, df.Age)
    df.Pclass = convert(Vector{Float64}, df.Pclass)
    df.SibSp = convert(Vector{Float64}, df.SibSp)

    onehotencoding!(df, :Embarked)
    # ------------

    inputs = [:Pclass, :Age, :SibSp] ∪ onehotdimensions(df, :Embarked)
    output = :Survived

    ranges = deriveranges(df, inputs)

    td = TrainingData("titanic", df; inputs, output)

    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)
    be = BoundaryExposer(td, modelsut; categoricals=[:Embarked])

    candidates = apply(be; timelimit=1, initial_candidates=5)
    df = todataframe(candidates, modelsut; output, categoricals=[:Embarked])

    for i in 1:nrow(df)
        @test df[i, :Survived] != df[i, :n_Survived]
    end

    plots(df, MLSqueeze.ranges(td); output)
end

