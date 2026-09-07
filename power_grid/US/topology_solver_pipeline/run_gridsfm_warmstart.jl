#!/usr/bin/env julia

"""
Run one topology-agnostic GridSFM `Pg + theta` primal initialization through
PowerModels/Ipopt and compare the converged state with the saved AC-OPF
reference in a `.pyg.json` scenario.

Usage:
  julia --project=. run_gridsfm_warmstart.jl \
    <transformed.json> <scenario.pyg.json> <prediction.json> <result.json>

The saved AC-OPF solution is comparison-only. It is never copied into the
PowerModels input. The interface is path-based and relies only on the ID maps
stored in each scenario, so the same runner can be used for training-like and
OOD graphs that follow the repository schema.
"""

using PowerModels
using Ipopt
using JuMP
using JSON3
using Printf
using Dates

PowerModels.silence()

const REQUIRED_PREDICTIONS = ("Pg", "theta")

_success(term) = any(s -> occursin(s, uppercase(term)),
                     ["LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED"])

function _load_dict(path::AbstractString)
    open(path) do io
        JSON3.read(io, Dict{String,Any})
    end
end

function _finite_vector(values, name)
    xs = Float64.(values)
    isempty(xs) && error("prediction $name is empty")
    all(isfinite, xs) || error("prediction $name contains NaN or Inf")
    xs
end

"Reset all four primal fields before applying the two-block learned seed."
function _reset_primal_start!(net)
    for b in values(get(net, "bus", Dict{String,Any}()))
        b["va"] = 0.0
        b["vm"] = 1.0
        pop!(b, "va_start", nothing)
        pop!(b, "vm_start", nothing)
    end
    for g in values(get(net, "gen", Dict{String,Any}()))
        g["pg"] = 0.0
        g["qg"] = 0.0
        pop!(g, "pg_start", nothing)
        pop!(g, "qg_start", nothing)
    end
    net
end

"Apply raw GridSFM active dispatch and voltage angle by exported source ID."
function apply_gridsfm_pg_theta!(net, pyg, prediction)
    preds = get(prediction, "predictions", nothing)
    preds isa AbstractDict || error("prediction file has no predictions object")
    for key in REQUIRED_PREDICTIONS
        haskey(preds, key) || error("prediction file is missing $key")
    end

    bus_ids = Int.(pyg["metadata"]["bus_id_map"])
    gen_ids = Int.(pyg["metadata"]["gen_id_map"])
    theta = _finite_vector(preds["theta"], "theta")
    pg = _finite_vector(preds["Pg"], "Pg")
    length(theta) == length(bus_ids) ||
        error("theta rows $(length(theta)) != bus_id_map rows $(length(bus_ids))")
    length(pg) == length(gen_ids) ||
        error("Pg rows $(length(pg)) != gen_id_map rows $(length(gen_ids))")

    _reset_primal_start!(net)

    for (id, value) in zip(bus_ids, theta)
        bus = get(net["bus"], string(id), nothing)
        bus === nothing && error("bus ID $id from bus_id_map is absent from network")
        bus["va"] = value
        bus["va_start"] = value
    end
    for (id, value) in zip(gen_ids, pg)
        gen = get(net["gen"], string(id), nothing)
        gen === nothing && error("generator ID $id from gen_id_map is absent from network")
        Int(get(gen, "gen_status", get(gen, "status", 1))) == 1 ||
            error("generator ID $id is inactive but appears in gen_id_map")
        gen["pg"] = value
        gen["pg_start"] = value
    end

    Dict(
        "bus_theta_count" => length(theta),
        "generator_pg_count" => length(pg),
        "voltage_magnitude_injected" => false,
        "reactive_generation_injected" => false,
    )
end

function _stats(errors)
    xs = Float64.(errors)
    Dict(
        "count" => length(xs),
        "mae" => sum(abs, xs) / length(xs),
        "rmse" => sqrt(sum(abs2, xs) / length(xs)),
        "max_abs" => maximum(abs, xs),
    )
end

_angle_error(actual, expected) = atan(sin(actual - expected), cos(actual - expected))

function _solution_differences(pyg, solution)
    bus_ids = Int.(pyg["metadata"]["bus_id_map"])
    gen_ids = Int.(pyg["metadata"]["gen_id_map"])
    bus_ref = pyg["solution"]["nodes"]["bus"]
    gen_ref = pyg["solution"]["nodes"]["generator"]

    length(bus_ids) == length(bus_ref) || error("reference bus row mismatch")
    length(gen_ids) == length(gen_ref) || error("reference generator row mismatch")

    theta = Float64[]
    vm = Float64[]
    pg = Float64[]
    qg = Float64[]
    bus_differences = Any[]
    generator_differences = Any[]
    for (row, id) in enumerate(bus_ids)
        solved = solution["bus"][string(id)]
        theta_difference = _angle_error(Float64(solved["va"]), Float64(bus_ref[row][1]))
        vm_difference = Float64(solved["vm"]) - Float64(bus_ref[row][2])
        push!(theta, theta_difference)
        push!(vm, vm_difference)
        push!(bus_differences, Dict(
            "bus_id" => id,
            "theta_radians_signed_difference" => theta_difference,
            "theta_degrees_signed_difference" => rad2deg(theta_difference),
            "vm_pu_signed_difference" => vm_difference,
        ))
    end
    for (row, id) in enumerate(gen_ids)
        solved = solution["gen"][string(id)]
        pg_difference = Float64(solved["pg"]) - Float64(gen_ref[row][1])
        qg_difference = Float64(solved["qg"]) - Float64(gen_ref[row][2])
        push!(pg, pg_difference)
        push!(qg, qg_difference)
        push!(generator_differences, Dict(
            "generator_id" => id,
            "pg_pu_signed_difference" => pg_difference,
            "qg_pu_signed_difference" => qg_difference,
        ))
    end

    errors = Dict(
        "bus_theta_radians" => _stats(theta),
        "bus_theta_degrees" => _stats(rad2deg.(theta)),
        "bus_vm_pu" => _stats(vm),
        "generator_pg_pu" => _stats(pg),
        "generator_qg_pu" => _stats(qg),
    )
    branch_differences = Any[]
    all_active_flow_differences = Float64[]
    all_reactive_flow_differences = Float64[]
    for (edge_type, ids_key) in (("ac_line", "ac_line_branch_ids"),
                                 ("transformer", "transformer_branch_ids"))
        reference = pyg["solution"]["edges"][edge_type]["features"]
        ids = Int.(pyg["metadata"][ids_key])
        length(ids) == length(reference) || error("reference $edge_type row mismatch")
        pt = Float64[]
        qt = Float64[]
        pf = Float64[]
        qf = Float64[]
        for (row, id) in enumerate(ids)
            solved = solution["branch"][string(id)]
            # PyG export convention is [pt, qt, pf, qf].
            differences = Float64[
                Float64(solved["pt"]) - Float64(reference[row][1]),
                Float64(solved["qt"]) - Float64(reference[row][2]),
                Float64(solved["pf"]) - Float64(reference[row][3]),
                Float64(solved["qf"]) - Float64(reference[row][4]),
            ]
            push!(pt, differences[1])
            push!(qt, differences[2])
            push!(pf, differences[3])
            push!(qf, differences[4])
            append!(all_active_flow_differences, (differences[1], differences[3]))
            append!(all_reactive_flow_differences, (differences[2], differences[4]))
            push!(branch_differences, Dict(
                "branch_id" => id,
                "edge_type" => edge_type,
                "pt_pu_signed_difference" => differences[1],
                "qt_pu_signed_difference" => differences[2],
                "pf_pu_signed_difference" => differences[3],
                "qf_pu_signed_difference" => differences[4],
            ))
        end
        errors[edge_type * "_pt_pu"] = _stats(pt)
        errors[edge_type * "_qt_pu"] = _stats(qt)
        errors[edge_type * "_pf_pu"] = _stats(pf)
        errors[edge_type * "_qf_pu"] = _stats(qf)
    end
    errors["all_branch_active_flows_pu"] = _stats(all_active_flow_differences)
    errors["all_branch_reactive_flows_pu"] = _stats(all_reactive_flow_differences)

    Dict(
        "aggregate_errors" => errors,
        "per_component_signed_differences" => Dict(
            "buses" => bus_differences,
            "generators" => generator_differences,
            "branches" => branch_differences,
        ),
    )
end

function run_warmstart(network_path, pyg_path, prediction_path)
    pyg = _load_dict(pyg_path)
    prediction = _load_dict(prediction_path)
    net = PowerModels.parse_file(network_path; import_all=false, validate=true)
    seed = apply_gridsfm_pg_theta!(net, pyg, prediction)

    optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => 10000,
        "tol" => 1e-6,
        "acceptable_tol" => 1e-4,
        "mu_init" => 1.0,
        "warm_start_bound_push" => 1.0,
        "sb" => "yes",
    )

    build_start = time()
    pm = PowerModels.instantiate_model(net, ACPPowerModel, PowerModels.build_opf)
    build_seconds = time() - build_start
    solve_start = time()
    result = PowerModels.optimize_model!(pm, optimizer=optimizer)
    solve_wall_seconds = time() - solve_start

    term = string(get(result, "termination_status", "UNKNOWN"))
    objective = try Float64(get(result, "objective", NaN)) catch; NaN end
    reference_objective = Float64(pyg["metadata"]["objective"])
    objective_signed_difference = objective - reference_objective
    objective_abs_difference = abs(objective - reference_objective)
    objective_percent_difference = 100.0 * objective_abs_difference /
                                   max(abs(reference_objective), 1.0)
    differences = haskey(result, "solution") ?
        _solution_differences(pyg, result["solution"]) : nothing
    recorded_presolve = try
        Float64(prediction["timing"]["end_to_end_seconds"])
    catch
        nothing
    end

    Dict(
        "generated_at_utc" => string(now(UTC)),
        "method" => "gridsfm_raw_pg_theta_primal_initialization",
        "inputs" => Dict(
            "network" => abspath(network_path),
            "scenario" => abspath(pyg_path),
            "prediction" => abspath(prediction_path),
        ),
        "seed" => seed,
        "ipopt_options" => Dict(
            "mu_init" => 1.0,
            "warm_start_bound_push" => 1.0,
            "tol" => 1e-6,
            "acceptable_tol" => 1e-4,
            "max_iter" => 10000,
        ),
        "reference" => Dict(
            "termination_status" => get(pyg["metadata"], "termination_status", nothing),
            "objective" => reference_objective,
            "solve_time_seconds" => get(pyg["metadata"], "solve_time_seconds", nothing),
        ),
        "warmstarted_acopf" => Dict(
            "termination_status" => term,
            "converged" => _success(term),
            "objective" => objective,
            "objective_signed_difference" => objective_signed_difference,
            "objective_absolute_difference" => objective_abs_difference,
            "objective_percent_difference" => objective_percent_difference,
            "model_build_seconds" => build_seconds,
            "solve_wall_seconds" => solve_wall_seconds,
            "solver_reported_solve_seconds" => get(result, "solve_time", nothing),
            "recorded_gridsfm_presolve_seconds" => recorded_presolve,
            "recorded_presolve_plus_solve_seconds" =>
                recorded_presolve === nothing ? nothing : recorded_presolve + solve_wall_seconds,
        ),
        "final_solution_differences_vs_reference" => differences,
    )
end

function main()
    length(ARGS) == 4 || error(
        "Usage: run_gridsfm_warmstart.jl <transformed.json> <scenario.pyg.json> " *
        "<prediction.json> <result.json>")
    network_path, pyg_path, prediction_path, output_path = ARGS
    for path in (network_path, pyg_path, prediction_path)
        isfile(path) || error("input not found: $path")
    end

    report = run_warmstart(network_path, pyg_path, prediction_path)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, report)
    end

    warm = report["warmstarted_acopf"]
    @printf("GRIDSFM_WARM status=%s objective=%.8f delta=%.8f (%.6f%%) solve=%.3fs\n",
            warm["termination_status"], warm["objective"],
            warm["objective_absolute_difference"],
            warm["objective_percent_difference"], warm["solve_wall_seconds"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
