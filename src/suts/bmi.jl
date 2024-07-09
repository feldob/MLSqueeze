function bmi(height::Float64, weight::Float64)
    if height < 0 || weight < 0
        #FIXME: for simplicity, dont consider exceptions here.
        #throw(DomainError("Height or Weight cannot be negative."))
        return -1
    end
    heigh_meters = height / 100 # Convert height from cm to meters
    bmivalue = round(weight / heigh_meters^2, digits = 1) # official standard expects 1 decimal after the comma
    return (bmivalue)
end

function bmi_classification(height::Float64, weight::Float64)
    bmivalue = bmi(height,weight)
    class = ""
    if bmivalue < 0
        #FOXME: for simplicity, dont consider exceptions here.
        #throw(DomainError(bmivalue, "BMI was negative. Check your inputs: $(height) cm; $(weight) kg"))
        class = "Err"
    elseif bmivalue < 18.5
        class = "Underweight"
    elseif bmivalue < 23
        class = "Normal"
    elseif bmivalue < 25
        class = "Overweight"
    elseif bmivalue < 30
        class = "Obese"
    else class = "Severely obese"
    end
    return class
end

const BMI_RANGES = [(-10,243), (-10,510)]
