using "LQI.jl"

mutable struct genetic
end

#Optimizer
function optimize(g::genetic, constraint::sample, system::plant)
    population = initDist(100, 5, 1e-2, length(B))
    population = [f(i, constraint, system) for i in population]
    order(population)

    sample_history = Vector{Vector{sample}}()
    average_function_val = Vector{Float64}()
    curr = 0
    prev = -1

    M = 10
    i = 1
    while abs(curr-prev) >= 1e-5
        println(i)
        push!(sample_history, population)
        push!(average_function_val, f(average(population), constraint, system).score)
        
        #New Generation
        parents = tournament(population, 20)
        zygotes = crossover(parents, 0.5)
        population = mutate(zygotes, M)
        M *= 0.9
        population = [f(j, constraint, system) for j in population]
        order(population)

        prev = curr
        curr = average_function_val[i]
        i += 1        
    end

    return f(average(population), constraint, system), sample_history, average_function_val
end


#Objective function

function average(population::Vector{sample})
    avg = sample(zeros(length(last(population).q)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in population
        avg = avg + 1/length(population)*i      
    end
    return avg
end

#Genetic Algorithms Functions
function order(X::Vector{sample})
    for j = 1:length(X)
        curr = X[j]
        i = j - 1
        while (i > 0) && (X[i] > curr)
            X[i+1] = X[i]
            i -= 1
        end
        X[i+1] = curr
    end
end

function tournament(population::Vector{sample}, k)
    parents = Vector{Vector{sample}}()
    n = length(population)
    for j = 1:n
        temp = rand(population, k)
        order(temp)
        push!(parents, temp[1:2])
    end
    return parents
end

function crossover(parents::Vector{Vector{sample}}, lambda::Float64)
    return [lambda*i[1] + (1-lambda)*i[2] for i in parents]
end

function mutate(zygotes::Vector{sample}, M)
    d = Normal(0.0, M)
    return [i + rand(d,length(i.q)) for i in zygotes]
end

