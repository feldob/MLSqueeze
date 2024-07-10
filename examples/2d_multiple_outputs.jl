using MLSqueeze, StatsPlots, DataFrames

# showcase with bmi classification

td = TrainingData(bmi_classification; ranges=BMI_RANGES, npoints = 1000)
be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))

iterations = 1000
initial_candidates = 50

# Regular MLSqueeze without diversity.
candidates = apply(be; iterations=initial_candidates, initial_candidates, optimizefordiversity=false)
df = todataframe(candidates, bmi_classification)
plot_nodiv = plots(df, MLSqueeze.ranges(td))[1]

# Regular MLSqueeze with diversity.
candidates = apply(be; iterations, initial_candidates)
df = todataframe(candidates, bmi_classification)
plot_div = plots(df, MLSqueeze.ranges(td))[1]

# run as one-vs-all... finds many more points.

# Regular MLSqueeze without diversity.
candidates = apply(be; iterations=50, initial_candidates, optimizefordiversity=false, strategy=OneVsAll("Normal"))
df = todataframe(candidates, bmi_classification)
plot_nodiv_1vsa = plots(df, MLSqueeze.ranges(td))[1]

# Regular MLSqueeze with diversity.
candidates = apply(be; iterations, initial_candidates=50, add_new=false, strategy=OneVsAll("Normal"))
df = todataframe(candidates, bmi_classification)
plot_div_1vsa = plots(df, MLSqueeze.ranges(td))[1]

plot(plot_nodiv, plot_div, plot_nodiv_1vsa, plot_div_1vsa, layout=(2, 2), size=(1200,800))