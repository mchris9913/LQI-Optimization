using ControlSystems
using LinearAlgebra
using Distributions
using Plots

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

mutable struct plant
    A::Matrix{Float64}
    B::Matrix{Float64}
    Bi::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
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

#Constructors
initConstraint() = sample(Vector{Float64}(), Inf, Inf, Inf, Inf, Inf, 0, 0, Inf)

#Control System
function tf2ss(num, den)
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

    return plant(Aa, Ba, Bi, Ca, D)
end

function f(s::sample, constraint::sample, system::plant)
    k = lqr(ss(system), s*s, 1.0)
    
    #time domain results
    timeresult = step(ss(system, k))
    info = stepinfo(timeresult)
    s.tr = info.risetime
    s.ts = info.settlingtime
    s.os = info.overshoot
    s.us = info.undershoot

    #frequency domain results
    wgm, gm, wpm, pm = margin(ss(system, k))
    s.gm = gm[1]
    s.pm = pm[1]

    #control effort
    s.ce = sum(abs.(k*timeresult[3]))

    if count(s, constraint) > 0
        penalties = 1000000000*count(s,constraint) + 1000000000*quadratic(s,constraint)
    else
        penalties = 1/1000000000*barrier(s,constraint)
    end
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

function barrier(s::sample, constraint::sample)
    pen = 0

    pen -= 1/(max((s.tr - constraint.tr),1e-4))
    pen -= 1/(max((s.ts - constraint.ts),1e-4))
    pen -= 1/(max((s.os - constraint.os),1e-4))
    pen -= 1/(max((s.us - constraint.us),1e-4))
    pen -= 1/(max((constraint.pm - s.pm),1e-4))
    pen -= 1/(max((constraint.gm - s.gm),1e-4))

    return pen
end

#Initial Distributions
function initDist(numSample::Int64, numBand::Int64, variance::Float64, states::Int64)
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
