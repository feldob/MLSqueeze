using Test
using MLSqueeze
using CSV
using DataFrames
using BlackBoxOptim
using Distances
using DecisionTree
using MLJ
using MLJDecisionTreeInterface

include("trainingdata_test.jl")
include("boundarysqueeze_test.jl")
include("boundaryexposer_test.jl")