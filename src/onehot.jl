function onehotencoding!(df::DataFrame, input_id::Symbol)

    # copy of the column
    input_id_col = df[:, input_id]
    
    # remove original column from dataframe
    select!(df, Not([input_id]))
    
    # get all the unique values from column
    values = unique(input_id_col)

    # add a column per each category
    for i in eachindex(values)
        v= values[i]
        df[!, "$(input_id)_$v"] = map(x -> x == v ? 1.0 : 0.0, input_id_col)
    end
end

# make only largest value a 1.0, otherwise 0.0
function repaironehotencoding!(df::DataFrame, col_ids::Vector{String})
    subdf = df[!, col_ids]
    for i in 1:nrow(subdf)
        row = collect(subdf[i,:])
        maxind = argmax(row)
        for j in 1:length(row)
            subdf[i,j] = j == maxind ? 1.0 : 0.0
        end
    end
end

function undoonehotencoding_internal!(df::DataFrame, input_id::Symbol, value_pos=2)
    col_ids = filter(x -> startswith(x, string(input_id)), names(df))

    repaironehotencoding!(df, col_ids)
    
    df[!, input_id] = Vector{String}(undef, nrow(df))
    for col_id in col_ids
        value = split(col_id, "_")[value_pos]
        for i in eachindex(df[!, input_id])
            if df[i, col_id] == 1.0
                df[i, input_id] = value
            end
        end
    end
    select!(df, Not(col_ids))
end

# do for both sides of the boundary
function undoonehotencoding!(df::DataFrame, input_id::Symbol)
    undoonehotencoding_internal!(df, input_id)
    undoonehotencoding_internal!(df, Symbol("n_$input_id"), 3)
end

onehotdimensions(df::DataFrame, inp::Symbol) = Symbol.(filter(n -> startswith(n, string(inp) * "_"), names(df)))