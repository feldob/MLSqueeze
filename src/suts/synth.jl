synth_func(x::Float64) = (x+2)*(x^2-4)
check_synth_valid(x::Float64,y::Float64) = y > synth_func(x)