import ControlSystems
import LinearAlgebra
import Distributions

#Sample
mutable struct sample
    q::Vector{Float64}
    score::Float64
    tr::Float64
    ts::Float64
    os::Float64
    us::Float64
    pm::Float64
    gm::Float64
    ce::Float64
end

#Adam optimizer
mutable struct adam
	alpha::Float64
	yv::Float64
	ys::Float64
	e::Float64
	k::Int64
	v::Vector{Float64}
	s::Vector{Float64}
end

#Overloaded operators
Base.:<(x::sample, y::sample) = x.score < y.score
Base.:>(x::sample, y::sample) = x.score > y.score
Base.:(>=)(x::sample, y::sample) = x.score >= y.score
Base.:(<=)(x::sample, y::sample) = x.score <= y.score
Base.:(==)(x::sample, y::sample) = x.score == y.score
Base.:*(x::sample, y::sample) = hcat(x.q)*transpose(hcat(x.q))
Base.:*(lambda::Float64, x::sample) = sample(lambda*x.q, x.score, x.tr, x.ts, x.os, x.us, x.pm, x.gm, x.ce)
Base.:+(x::sample, y::sample) = sample(x.q + y.q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
Base.:+(x::sample, v::Vector{Float64}) = sample(x.q + v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
Base.:-(x::sample, v::Vector{Float64}) = sample(x.q - v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#Control System
function plant(num, den)
    sys = ss(tf(num, den))
    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    Aa = [A [0;0]; -C 0]
    Ba = [B; 0]
    Ca = [C 0]

    Bi = zeros(length(B))
    Bi = hcat([Bi; 1])

    return Aa, Ba, Bi, Ca, D
end

#Optimizer
function optimize(constraint::sample, A, B, Bi, C, D)
    population = initDist(100, 1e-2, length(B))
    population = [f(i, constraint, A, B, Bi, C, D) for i in population]
    order(population)

    sample_history = Vector{Vector{sample}}()
    average_function_val = Vector{Float64}()
    curr = 0
    prev = -1

    M = 10
    i = 1
    while abs(curr-prev) >= 1
        println(i)
        push!(sample_history, population)
        push!(average_function_val, f(average(population), constraint, A, B, Bi, C, D).score)
        
        #New Generation
        parents = tournament(population, 20)
        zygotes = crossover(parents, 0.5)
        population = mutate(zygotes, M)
        M *= 0.9
        population = [f(j, constraint, A, B, Bi, C, D) for j in population]
        order(population)

        prev = curr
        curr = average_function_val[i]
        i += 1        
    end

    return f(average(population), constraint, A, B, Bi, C, D), sample_history, average_function_val
end

function optimize(a::adam, s::sample, constraint::sample, A, B, Bi, C, D)
    curr = f(s, constraint, A, B, Bi, C, D).score
    prev = 0
    average_function_val = Vector{Float64}()

	a.alpha = 1e-4
	a.v = zeros(length(B))
	a.s = zeros(length(B))
	a.k = 0

    i = 1
    while abs(curr-prev) >= 1e-20
        println(i)
        push!(average_function_val, curr)

        s = step!(alpha, s, grad(s, constraint, A, B, Bi, C, D))
        s = f(s, constraint, A, B, Bi, C, D)
        prev = curr;
        curr = s.score

        if i % 1000 == 0
            a.alpha /= 10
        end
        i += 1
    end

    return s, average_function_val
end

#Objective function
initConstraint() = sample(Vector{Float64}(), Inf, Inf, Inf, Inf, Inf, 0, 0, Inf)

function f(s::sample, constraint::sample, A, B, Bi, C, D)
    k = lqr(ss(A,B,C,D), s*s, 1.0)
    
    #time domain results
    timeresult = step(ss(A-B*k,Bi,C,D))
    info = stepinfo(timeresult)
    s.tr = info.risetime
    s.ts = info.settlingtime
    s.os = info.overshoot
    s.us = info.undershoot

    #frequency domain results
    wgm, gm, wpm, pm = margin(ss(A-B*k, Bi, C, D))
    s.gm = gm[1]
    s.pm = pm[1]

    #control effort
    s.ce = sum(abs.(k*timeresult[3]))

    penalties = 1000000000*count(s,constraint) + 1000000000*quadratic(s,constraint)
    s.score = s.tr + s.ts + s.os + s.us + s.ce + exp(-s.gm) + exp(-s.pm) + penalties
    return s
end

function count(s::sample, constraint::sample)
    pen = 0

    pen += (s.tr > constraint.tr) ? 1 : 0
    pen += (s.ts > constraint.ts) ? 1 : 0
    pen += (s.os > constraint.os) ? 1 : 0
    pen += (s.us > constraint.us) ? 1 : 0
    pen += (constraint.pm > s.pm) ? 1 : 0
    pen += (constraint.gm > s.gm) ? 1 : 0

    return pen
end

function quadratic(s::sample, constraint::sample)
    pen = 0

    pen += max(s.tr - constraint.tr, 0)^2
    pen += max(s.ts - constraint.ts, 0)^2
    pen += max(constraint.pm - s.pm, 0)^2
    pen += max(constraint.gm - s.gm, 0)^2
    pen += max(s.os - constraint.os, 0)^2
    pen += max(s.us - constraint.os, 0)^2

    return pen
end

function average(population::Vector{sample})
    avg = sample(zeros(length(last(population).q)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in population
        avg = avg + 1/length(population)*i      
    end
    return avg
end

#Initial Distributions
function initDist(numSample::Int64, numBand::Float64, variance::Float64, states::Int64)
    dist = Vector{Vector{Float64}}()
    
    numPerBand = numSample / numBand
    for i = 1:numBand
        d = Normal(0.0, variance)
        for j = 1:numPerBand
            push!(dist, abs.(rand(d, states)))
        end
        variance *= 10
    end

    return [map(i) for i in dist]
end

function map(v::Vector{Float64})
    return sample(v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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

#adam
function grad(s::sample, constraint::sample, A, B, Bi, C, D)
    grad = Vector{Float64}()
    curr = f(s, constraint, A, B, Bi, C, D)

    h = 1e-10
    stepsize = h*Matrix{Float64}(I, size(A))

    for i = 1:length(B)
        push!(grad, (f(s + stepsize[:,i], constraint, A, B, Bi, C, D).score - curr.score)/h)
    end

    return grad
end

initAdam() = adam(0.0, 0.0, 0.0, 0.0, 0, Vector{Float64}(), Vector{Float64}())

function step!(a::adam, x::sample, grad::Vector{Float64})
	a.v = a.yv*a.v + (1-a.yv)*grad
	a.s = a.ys*a.s + (1-a.ys)*grad.*grad
	
	a.k += 1
	v = a.v ./ (1-yv^k)
	s = a.s ./ (1-ys^k)
	return x - a.alpha*v ./ (sqrt.(s) .+ a.e)
end
