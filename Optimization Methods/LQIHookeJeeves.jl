mutable struct HookeJeeves
    h::Float64
    p::Float64
end

function optimize(hj::HookeJeeves, controller::sample, constraint::sample, system::plant)
    controller_score = Vector{Float64}()
    controller_history = Vector{sample}()
    controller = f(controller, constraint, system)

    i = 1
    while hj.h > 1e-10
        println(i)

        push!(controller_score, controller.score)
        push!(controller_history, controller)
        controller = bestStep(hj, controller, constraint, system)
        
        i += 1
    end

    return controller
end

function bestStep(hj::HookeJeeves, controller::sample, constraint::sample, system::plant)
    stepsize = hj.h*Matrix{Float64}(I, size(system.A))

    bestScore = controller.score
    bestController = controller
    for i = 1:length(system.B)
        test = f(controller + stepsize[:,i], constraint, system)
        if test.score <= bestScore
            bestController = test
            bestScore = test.score            
        end

        test = f(controller - stepsize[:,i], constraint, system)
        if test.score <= bestScore
            bestController = test
            bestScore = test.score            
        end
    end

    if bestController.q == controller.q
        hj.h *= hj.p
    end

    return bestController
end