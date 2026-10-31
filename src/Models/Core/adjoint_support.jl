# Adjoint support check for the concrete model. Driven schedule and capture
# methods live beside DrivenSimulation in ../Runner/adjoint_schedule.jl, whose
# `require_adjoint_support` docstring also documents this method.
function Adjoints.require_adjoint_support(model::TransportModel;
                                          plan::StepPlan = _MODEL_STEP_PLAN)
    require_adjoint_support(plan, step_operators(model))
    return model
end
