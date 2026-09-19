#!/usr/bin/env julia

"""Paired cold/DC-warm/GridSFM-warm AC-OPF benchmark over saved predictions."""

include("run_gridsfm_warmstart.jl")
using LinearAlgebra

BLAS.set_num_threads(1)

function _optimizer(; approximate=false)
    attributes = Pair{String,Any}[
        "print_level" => 0,
        "max_iter" => 10000,
        "tol" => 1e-6,
        "acceptable_tol" => 1e-4,
        "sb" => "yes",
    ]
    if approximate
        append!(attributes, [
            "mu_init" => 1.0,
            "warm_start_bound_push" => 1.0,
        ])
    end
    optimizer_with_attributes(Ipopt.Optimizer, attributes...)
end

function _reference_document(pyg, solution)
    bus_ids = Int.(pyg["metadata"]["bus_id_map"])
    gen_ids = Int.(pyg["metadata"]["gen_id_map"])
    nodes = Dict(
        "bus" => Any[[Float64(solution["bus"][string(id)]["va"]),
                       Float64(solution["bus"][string(id)]["vm"])] for id in bus_ids],
        "generator" => Any[[Float64(solution["gen"][string(id)]["pg"]),
                             Float64(solution["gen"][string(id)]["qg"])] for id in gen_ids],
    )
    edges = Dict{String,Any}()
    for (edge_type, ids_key) in (("ac_line", "ac_line_branch_ids"),
                                 ("transformer", "transformer_branch_ids"))
        ids = Int.(pyg["metadata"][ids_key])
        edges[edge_type] = Dict("features" => Any[
            [Float64(solution["branch"][string(id)]["pt"]),
             Float64(solution["branch"][string(id)]["qt"]),
             Float64(solution["branch"][string(id)]["pf"]),
             Float64(solution["branch"][string(id)]["qf"])] for id in ids
        ])
    end
    Dict("metadata" => pyg["metadata"], "solution" => Dict("nodes" => nodes, "edges" => edges))
end

function _solve_ac(net; approximate=false)
    build_start = time()
    pm = PowerModels.instantiate_model(net, ACPPowerModel, PowerModels.build_opf)
    build_seconds = time() - build_start
    solve_start = time()
    result = PowerModels.optimize_model!(pm, optimizer=_optimizer(approximate=approximate))
    solve_wall_seconds = time() - solve_start
    Dict(
        "termination_status" => string(get(result, "termination_status", "UNKNOWN")),
        "converged" => _success(string(get(result, "termination_status", "UNKNOWN"))),
        "objective" => try Float64(get(result, "objective", NaN)) catch; NaN end,
        "ac_model_build_seconds" => build_seconds,
        "ac_solve_wall_seconds" => solve_wall_seconds,
        "ac_solver_reported_seconds" => get(result, "solve_time", nothing),
        "ac_ipopt_iterations" => _ipopt_iterations(pm),
    ), result
end

function _apply_dc_seed!(net, dc_solution)
    _reset_primal_start!(net)
    for (id, solved) in dc_solution["bus"]
        value = Float64(solved["va"])
        net["bus"][id]["va"] = value
        net["bus"][id]["va_start"] = value
    end
    for (id, solved) in dc_solution["gen"]
        value = Float64(solved["pg"])
        net["gen"][id]["pg"] = value
        net["gen"][id]["pg_start"] = value
    end
    net
end

"Paper-described cold state: flat voltage/angle and midpoint active dispatch."
function _apply_paper_cold_start!(net)
    _reset_primal_start!(net)
    for bus in values(get(net, "bus", Dict{String,Any}()))
        bus["va_start"] = 0.0
        bus["vm_start"] = 1.0
    end
    for gen in values(get(net, "gen", Dict{String,Any}()))
        midpoint = (Float64(gen["pmin"]) + Float64(gen["pmax"])) / 2.0
        gen["pg"] = midpoint
        gen["pg_start"] = midpoint
        gen["qg_start"] = 0.0
    end
    net
end

function _dc_seed(base_net)
    start = time()
    dc_net = deepcopy(base_net)
    _reset_primal_start!(dc_net)
    dc_pm = PowerModels.instantiate_model(dc_net, DCPPowerModel, PowerModels.build_opf)
    dc_result = PowerModels.optimize_model!(dc_pm, optimizer=_optimizer())
    term = string(get(dc_result, "termination_status", "UNKNOWN"))
    _success(term) || error("DC-OPF seed solve failed with $term")
    target = deepcopy(base_net)
    _apply_dc_seed!(target, dc_result["solution"])
    elapsed = time() - start
    target, Dict(
        "seconds" => elapsed,
        "termination_status" => term,
        "solver_reported_seconds" => get(dc_result, "solve_time", nothing),
        "ipopt_iterations" => _ipopt_iterations(dc_pm),
        "objective" => get(dc_result, "objective", nothing),
    )
end

function _run_triplet(base_net, pyg, prediction, repetition;
                      retain_components=true, paper_cold=false)
    cold_net = deepcopy(base_net)
    paper_cold ? _apply_paper_cold_start!(cold_net) : _reset_primal_start!(cold_net)
    cold, cold_result = _solve_ac(cold_net)
    cold["arm"] = "cold"
    cold["repetition"] = repetition
    cold["seed_preparation_seconds"] = 0.0
    cold["seeded_total_seconds"] = cold["ac_solver_reported_seconds"]
    cold["workflow_e2e_seconds"] = cold["ac_model_build_seconds"] + cold["ac_solve_wall_seconds"]
    cold["objective_signed_difference_vs_matched_cold"] = 0.0
    cold["objective_percent_difference_vs_matched_cold"] = 0.0

    matched_reference = _reference_document(pyg, cold_result["solution"])

    gridsfm_net = deepcopy(base_net)
    mapping_start = time()
    seed_metadata = apply_gridsfm_pg_theta!(gridsfm_net, pyg, prediction)
    mapping_seconds = time() - mapping_start
    prediction_timing = prediction["timing"]
    # Historical artifacts used `end_to_end_seconds` for this model-resident
    # boundary. New artifacts name it explicitly and reserve E2E for the
    # externally observed cold CLI request.
    recorded_inference = Float64(
        haskey(prediction_timing, "resident_request_seconds") ?
        prediction_timing["resident_request_seconds"] :
        prediction_timing["end_to_end_seconds"]
    )
    gridsfm_presolve = recorded_inference + mapping_seconds
    gridsfm, gridsfm_result = _solve_ac(gridsfm_net; approximate=true)
    gridsfm["arm"] = "gridsfm_warm"
    gridsfm["repetition"] = repetition
    gridsfm["seed"] = seed_metadata
    gridsfm["recorded_gridsfm_inference_seconds"] = recorded_inference
    gridsfm["current_seed_mapping_seconds"] = mapping_seconds
    gridsfm["seed_preparation_seconds"] = gridsfm_presolve
    gridsfm["seeded_total_seconds"] = gridsfm_presolve + gridsfm["ac_solver_reported_seconds"]
    gridsfm["workflow_e2e_seconds"] = gridsfm_presolve + gridsfm["ac_model_build_seconds"] + gridsfm["ac_solve_wall_seconds"]

    dc_net, dc_presolve = _dc_seed(base_net)
    dc, dc_result = _solve_ac(dc_net; approximate=true)
    dc["arm"] = "dc_warm"
    dc["repetition"] = repetition
    dc["dc_presolve"] = dc_presolve
    dc["seed_preparation_seconds"] = dc_presolve["seconds"]
    dc["seeded_total_seconds"] = dc_presolve["seconds"] + dc["ac_solver_reported_seconds"]
    dc["workflow_e2e_seconds"] = dc_presolve["seconds"] + dc["ac_model_build_seconds"] + dc["ac_solve_wall_seconds"]

    cold_objective = cold["objective"]
    for (record, result) in ((gridsfm, gridsfm_result), (dc, dc_result))
        signed = record["objective"] - cold_objective
        record["objective_signed_difference_vs_matched_cold"] = signed
        record["objective_percent_difference_vs_matched_cold"] =
            100.0 * abs(signed) / max(abs(cold_objective), 1.0)
        if haskey(result, "solution")
            differences = _solution_differences(matched_reference, result["solution"])
            if !retain_components
                pop!(differences, "per_component_signed_differences", nothing)
            end
            record["final_solution_differences_vs_matched_cold"] = differences
        end
    end
    Dict("cold" => cold, "gridsfm_warm" => gridsfm, "dc_warm" => dc)
end

function _write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, value)
    end
end

function _scenario_paths(root, sample_id)
    Dict(
        "network" => joinpath(root, "scenarios", sample_id * "_transformed.json"),
        "scenario" => joinpath(root, "scenarios", sample_id * ".pyg.json"),
        "prediction" => joinpath(root, "predictions", sample_id * "_prediction.json"),
    )
end

function main_5x5()
    2 <= length(ARGS) <= 3 || error(
        "Usage: run_warmstart_5x5.jl <artifact-root> <output-dir> [repetitions]")
    root, output_dir = abspath.(ARGS[1:2])
    repetitions = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 5
    prediction_files = sort(filter(f -> endswith(f, "_prediction.json"),
                                   readdir(joinpath(root, "predictions"))))
    sample_ids = replace.(prediction_files, "_prediction.json" => "")
    length(sample_ids) == 25 || error("expected 25 predictions, found $(length(sample_ids))")

    config = Dict(
        "generated_at_utc" => string(now(UTC)),
        "corpus" => "seed42_5x5_mixed_combination_perturbations",
        "sample_ids" => sample_ids,
        "repetitions" => repetitions,
        "unmeasured_warmups_per_scenario" => 1,
        "arms" => ["cold", "gridsfm_warm", "dc_warm"],
        "threads" => Dict("julia" => Threads.nthreads(), "blas" => BLAS.get_num_threads()),
        "gridsfm_presolve_note" => "Saved inference E2E plus current Julia ID mapping; inference is not re-run in this batch.",
        "approximate_start_options" => Dict("mu_init" => 1.0, "warm_start_bound_push" => 1.0),
    )
    _write_json(joinpath(output_dir, "config.json"), config)

    for (scenario_index, sample_id) in enumerate(sample_ids)
        paths = _scenario_paths(root, sample_id)
        output_path = joinpath(output_dir, sample_id * "_runs.json")
        if isfile(output_path)
            existing = _load_dict(output_path)
            if length(get(existing, "measured_runs", [])) == repetitions
                @printf("SKIP %d/%d %s complete\n", scenario_index, length(sample_ids), sample_id)
                continue
            end
        end
        @printf("SCENARIO %d/%d %s\n", scenario_index, length(sample_ids), sample_id)
        pyg = _load_dict(paths["scenario"])
        prediction = _load_dict(paths["prediction"])
        base_net = PowerModels.parse_file(paths["network"]; import_all=false, validate=true)

        # Compile and warm all three formulations without using the observation.
        _run_triplet(base_net, pyg, prediction, 0; retain_components=false)
        measured = Any[]
        report = Dict(
            "sample_id" => sample_id,
            "paths" => paths,
            "perturbation_mode" => "mixed_combination",
            "measured_runs" => measured,
        )
        for repetition in 1:repetitions
            push!(measured, _run_triplet(base_net, pyg, prediction, repetition))
            _write_json(output_path, report)
            @printf("  REP %d/%d cold=%.3fs gridsfm_total=%.3fs dc_total=%.3fs\n",
                    repetition, repetitions,
                    measured[end]["cold"]["ac_solver_reported_seconds"],
                    measured[end]["gridsfm_warm"]["seeded_total_seconds"],
                    measured[end]["dc_warm"]["seeded_total_seconds"])
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_5x5()
end
