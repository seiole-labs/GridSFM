#!/usr/bin/env julia
# Strict source-base → exported-PyG re-solve audit.  This is deliberately a
# verifier, not a repair tool: it never changes the source or relaxes a model.

include("solve_pyg_json.jl")
using JSON3
using Printf
using Dates

const STATE_MAX_TOL = 1e-3       # pu for Vm/P/Q/flows; radians for Va
const OBJECTIVE_REL_TOL = 1e-3   # 0.1 percent

_success(term) = any(s -> occursin(s, uppercase(term)),
                     ["LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED"])

function _stats(values)
    xs = Float64.(values)
    Dict("mae" => sum(abs, xs) / length(xs), "max_abs" => maximum(abs, xs),
         "count" => length(xs))
end

function _component_errors(pyg, sol)
    bid = Int.(pyg["metadata"]["bus_id_map"])
    gid = Int.(pyg["metadata"]["gen_id_map"])
    bus_pred = pyg["solution"]["nodes"]["bus"]
    gen_pred = pyg["solution"]["nodes"]["generator"]
    va = Float64[]; vm = Float64[]; pg = Float64[]; qg = Float64[]
    for (i, id) in enumerate(bid)
        push!(va, Float64(sol["bus"][string(id)]["va"]) - Float64(bus_pred[i][1]))
        push!(vm, Float64(sol["bus"][string(id)]["vm"]) - Float64(bus_pred[i][2]))
    end
    for (i, id) in enumerate(gid)
        push!(pg, Float64(sol["gen"][string(id)]["pg"]) - Float64(gen_pred[i][1]))
        push!(qg, Float64(sol["gen"][string(id)]["qg"]) - Float64(gen_pred[i][2]))
    end
    errors = Dict("bus_va_radians" => _stats(va), "bus_vm_pu" => _stats(vm),
                  "generator_pg_pu" => _stats(pg), "generator_qg_pu" => _stats(qg))
    for (edge_type, ids_key) in (("ac_line", "ac_line_branch_ids"),
                                 ("transformer", "transformer_branch_ids"))
        pred = pyg["solution"]["edges"][edge_type]["features"]
        ids = Int.(pyg["metadata"][ids_key])
        pt = Float64[]; qt = Float64[]; pf = Float64[]; qf = Float64[]
        for (i, id) in enumerate(ids)
            b = sol["branch"][string(id)]
            # PyG export convention is [pt, qt, pf, qf].
            push!(pt, Float64(b["pt"]) - Float64(pred[i][1]))
            push!(qt, Float64(b["qt"]) - Float64(pred[i][2]))
            push!(pf, Float64(b["pf"]) - Float64(pred[i][3]))
            push!(qf, Float64(b["qf"]) - Float64(pred[i][4]))
        end
        errors[edge_type * "_pt_pu"] = _stats(pt)
        errors[edge_type * "_qt_pu"] = _stats(qt)
        errors[edge_type * "_pf_pu"] = _stats(pf)
        errors[edge_type * "_qf_pu"] = _stats(qf)
    end
    errors
end

function main()
    length(ARGS) == 3 || error("Usage: audit_pyg_roundtrip.jl <source.m> <export.pyg.json> <report.json>")
    source, pyg_path, out = ARGS
    pyg = open(pyg_path) do io JSON3.read(io, Dict{String,Any}) end
    net = build_net_from_pyg(source, pyg)
    t0 = time()
    pm = PowerModels.instantiate_model(net, ACPPowerModel, PowerModels.build_opf)
    result = PowerModels.optimize_model!(pm, optimizer=optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "max_iter" => 10000, "tol" => 1e-6,
        "acceptable_tol" => 1e-4, "sb" => "yes"))
    elapsed = time() - t0
    term = string(get(result, "termination_status", "UNKNOWN"))
    obj = Float64(get(result, "objective", NaN))
    expected = Float64(pyg["metadata"]["objective"])
    rel_obj = abs(obj - expected) / max(abs(expected), 1.0)
    errors = _component_errors(pyg, result["solution"])
    max_state = maximum(Float64(v["max_abs"]) for v in values(errors))
    passed = _success(term) && rel_obj <= OBJECTIVE_REL_TOL && max_state <= STATE_MAX_TOL
    report = Dict(
        "purpose" => "Strict source-base to exported-PyG re-solve comparison; no relaxation or source modification.",
        "generated_at_utc" => string(now(UTC)),
        "source_path" => abspath(source), "pyg_path" => abspath(pyg_path),
        "tolerances" => Dict("objective_relative" => OBJECTIVE_REL_TOL,
                               "state_and_flow_max_abs" => STATE_MAX_TOL),
        "source_export" => Dict("termination_status" => pyg["metadata"]["termination_status"],
                                "objective" => expected,
                                "solve_time_seconds" => get(pyg["metadata"], "solve_time_seconds", nothing)),
        "pyg_resolve" => Dict("termination_status" => term, "objective" => obj,
                               "solve_time_seconds" => elapsed,
                               "objective_relative_difference" => rel_obj),
        "errors" => errors, "max_state_or_flow_abs_error" => max_state,
        "passed" => passed,
    )
    mkpath(dirname(out)); open(out, "w") do io JSON3.pretty(io, report) end
    @printf("ROUNDTRIP status=%s obj_rel=%.8g max_state_flow=%.8g passed=%s\n",
            term, rel_obj, max_state, passed)
    exit(passed ? 0 : 1)
end

main()
