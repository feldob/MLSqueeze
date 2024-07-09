@testset "test squeeze" begin
    ranges = [(-10,10), (-10,10)]

    td = TrainingData(check_synth_valid; ranges)

    # manually decide points on either side.
    good = [0.0,5.0]
    bad = [5.0,0.0]

    @test check_synth_valid(good...) == true
    @test check_synth_valid(bad...) == false
    
    squeeze = BoundarySqueeze(ranges; Delta = [1e-1, 1e-6]) # include different deltas for acceptance
    bc = apply(squeeze, check_synth_valid, good, bad)

    o1, o2 = check_synth_valid(left(bc)...), check_synth_valid(right(bc)...)
    @test o1 != o2                                              # different outputs
    @test isminimal(squeeze, bc)
end