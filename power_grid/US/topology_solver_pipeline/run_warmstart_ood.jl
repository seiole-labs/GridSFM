#!/usr/bin/env julia

"""Run the paired paper-style warm-start benchmark from an explicit manifest."""

include("run_warmstart_5x5.jl")
include("solve_pyg_json.jl")

function main_ood()
    2 <= length(ARGS) <= 3 || error(
        "Usage: run_warmstart_ood.jl <manifest.json> <output-dir> [repetitions]")
    manifest_path, output_dir = abspath.(ARGS[1:2])
    repetitions = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 5
    manifest = _load_dict(manifest_path)
    cases = manifest["cases"]
    repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    resolve_path(path) = isabspath(path) ? path : normpath(joinpath(repo_root, path))

    _write_json(joinpath(output_dir, "config.json"), Dict(
        "generated_at_utc" => string(now(UTC)),
        "manifest" => manifest_path,
        "sample_ids" => [case["sample_id"] for case in cases],
        "repetitions" => repetitions,
        "unmeasured_warmups_per_case" => 1,
        "cold_initialization" => "paper_described_vm_1_va_0_pg_capacity_midpoint_qg_0",
        "paper_timing_boundary" => "cold AC solver; GridSFM seeded AC solver; DC presolve plus seeded AC solver",
        "threads" => Dict("julia" => Threads.nthreads(), "blas" => BLAS.get_num_threads()),
    ))

    for (case_index, case) in enumerate(cases)
        sample_id = String(case["sample_id"])
        output_path = joinpath(output_dir, sample_id * "_runs.json")
        if isfile(output_path)
            existing = _load_dict(output_path)
            if length(get(existing, "measured_runs", [])) == repetitions
                @printf("SKIP %d/%d %s complete\n", case_index, length(cases), sample_id)
                continue
            end
        end
        source = resolve_path(String(case["source"]))
        scenario = resolve_path(String(case["scenario"]))
        prediction_path = resolve_path(String(case["prediction"]))
        for path in (source, scenario, prediction_path)
            isfile(path) || error("missing input: $path")
        end
        @printf("OOD %d/%d %s gate=%s\n", case_index, length(cases), sample_id, case["gate"])
        pyg = _load_dict(scenario)
        prediction = _load_dict(prediction_path)
        base_net = build_net_from_pyg(source, pyg)
        _reset_primal_start!(base_net)

        _run_triplet(base_net, pyg, prediction, 0;
                     retain_components=false, paper_cold=true)
        measured = Any[]
        report = Dict(
            "sample_id" => sample_id,
            "gate" => case["gate"],
            "paths" => Dict("source" => source, "scenario" => scenario,
                            "prediction" => prediction_path),
            "measured_runs" => measured,
        )
        for repetition in 1:repetitions
            push!(measured, _run_triplet(base_net, pyg, prediction, repetition;
                                         retain_components=false, paper_cold=true))
            _write_json(output_path, report)
            @printf("  REP %d/%d cold=%.3fs grid_ac=%.3fs dc_total=%.3fs\n",
                    repetition, repetitions,
                    measured[end]["cold"]["ac_solver_reported_seconds"],
                    measured[end]["gridsfm_warm"]["ac_solver_reported_seconds"],
                    measured[end]["dc_warm"]["seeded_total_seconds"])
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_ood()
end
