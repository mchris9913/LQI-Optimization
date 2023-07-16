mutable struct adam
	alpha::Float64
	yv::Float64
	ys::Float64
	e::Float64
	k::Int64
	v::Vector{Float64}
	s::Vector{Float64}
end

function optimize(a::adam, s::sample, constraint::sample, system)
    curr = f(s, constraint, system).score
    prev = 0
    average_function_val = Vector{Float64}()

	a.alpha = 1e-6
	a.v = zeros(length(B))
	a.s = zeros(length(B))
	a.k = 0

    i = 1
    while abs(curr-prev) >= 1e-20
        println(i)
        push!(average_function_val, curr)

        g = grad(s, constraint, system
        d = step!(a, s, g)
        a.alpha = bls(d, g, s, constraint, system)
        s = s + a.alpha*d

        s = f(s, constraint, system)
        prev = curr;
        curr = s.score

        i += 1
    end

    return s, average_function_val
end

function grad(s::sample, constraint::sample, system::plant)
    grad = Vector{Float64}()
    curr = f(s, constraint, system)

    h = 1e-10
    stepsize = h*Matrix{Float64}(I, size(system.A))

    for i = 1:length(system.B)
        push!(grad, (f(s + stepsize[:,i], constraint, system).score - curr.score)/h)
    end

    return grad
end

initAdam() = adam(0.0, 0.0, 0.0, 0.0, 0, Vector{Float64}(), Vector{Float64}())

function step!(a::adam, x::sample, grad::Vector{Float64})
	a.v = a.yv*a.v + (1-a.yv)*grad
	a.s = a.ys*a.s + (1-a.ys)*grad.*grad
	
	a.k += 1
	v = a.v ./ (1-a.yv^a.k)
	s = a.s ./ (1-a.ys^a.k)
	return v ./ (sqrt.(s) .+ a.e)
end

function bls(d::Vector{Float64}, g::Vector{Float64}, s::sample, constraint::sample, system::plant; alpha = 1, p = 0.5, b = 1e-4)
    while f(s+alpha*d, constraint, plant).score > s.score + b*alpha*dot(g,d)
        alpha *= p
    end
    return alpha
end