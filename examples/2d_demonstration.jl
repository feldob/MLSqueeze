using MLSqueeze, StatsPlots, DataFrames

# synth function defined under src/suts/synth.jl
#synth_func(x::Float64) = (x+2)*(x^2-4)
#check_synth_valid(x::Float64,y::Float64) = y > synth_func(x)

x = range(-12, 12, 1000)
y = synth_func.(x)

plot(x, y, xlims=(-8, 8), ylims=(-12, 12), xticks=false, yticks=false, xaxis=false, yaxis=false, seriestype=:line, label="ground truth", legend=:topleft)

annotate!(0, -13, Plots.text("argument 1 (x)", 10, :dark))
xlabel!(" ")
annotate!(-8.5, 0, Plots.text("argument 2 (y)", 10, :dark, rotation = 90 ))
ylabel!(" ")

using CSV
td = TrainingData(check_synth_valid; ranges=[(-12,12), (-12,12)], npoints = 300)

df_n, df_p = groupby(td.df, :output)

plot!(df_p.x, df_p.y, seriestype=:scatter, alpha=.8, markershape=:xcross, color=:green, label="training data (1s)")
plot!(df_n.x, df_n.y, seriestype=:scatter, alpha=.8, markershape=:cross, color=:orange, label="training data (0s)")

using DecisionTree # with model
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
be = BoundaryExposer(td, modelsut) # instantiate search alg

candidates = apply(be; iterations=10000, initial_candidates=30, optimizefordiversity=false) # search and collect candidates
df_ground_truth = todataframe(candidates, modelsut; output=:output)

plot!(df_ground_truth.x, df_ground_truth.y, color=:black, markersize=2, seriestype=:scatter, label="decision tree")

# squeeze
candidates = apply(be; iterations=2000, initial_candidates=20, add_new=false) # search and collect candidates
df = todataframe(candidates, modelsut; output=:output)
plot!(df.x, df.y, seriestype=:scatter, markersize=6, markerstrokewidth=0.5, marker=:circle, color=:red, label="boundary candidates")