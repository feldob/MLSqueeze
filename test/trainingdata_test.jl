@testset "test defaultranges" begin
    dr = defaultranges(bmi)
    @test dr[1] == (typemin(Float64), typemax(Float64))
    @test dr[2] == (typemin(Float64), typemax(Float64))
end

@testset "test TrainingData" begin
    td = TrainingData(bmi; ranges=BMI_RANGES, npoints = 30)
    @test sutname(td) == "bmi"
    @test npoints(td) == 30
    @test !classproblem(td)
end

@testset "test TrainingData classification" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES, npoints = 30)
    @test sutname(td) == "bmi_classification"
    @test npoints(td) == 30
    @test classproblem(td)
end

@testset "plotting TrainingData test" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES, npoints = 30) 
    plots(td)
end

@testset "TrainingData import existing classification" begin
    df = CSV.read("../data/Iris.csv", DataFrame)
    
    inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
    output = :Species

    td = TrainingData("iris", df; inputs, output)
    plts = plots(td)
    @test length(plts) == 6
end