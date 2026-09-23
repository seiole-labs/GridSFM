#!/usr/bin/env julia
# PROTOTYPE: solve one reconstructed OPFData network with strict AC-OPF.

using PowerModels, Ipopt, JuMP, JSON3

PowerModels.silence()

if length(ARGS) != 2
    println(stderr, "usage: julia prototype_solve_powermodels_case.jl <network.json> <result.json>")
    exit(2)
end

input_path, output_path = ARGS
network = PowerModels.parse_file(input_path; import_all=false, validate=true)
solver = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "sb" => "yes",
    "max_iter" => 10_000,
    "tol" => 1e-6,
    "acceptable_tol" => 1e-4,
)

pm = PowerModels.instantiate_model(network, ACPPowerModel, PowerModels.build_opf)
result = PowerModels.optimize_model!(pm, optimizer=solver)

open(output_path, "w") do io
    JSON3.pretty(io, result)
end

println("termination_status=$(get(result, \"termination_status\", \"UNKNOWN\"))")
println("objective=$(get(result, \"objective\", NaN))")
