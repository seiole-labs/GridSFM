#!/usr/bin/env julia
# Audit raw PGLib/MATPOWER cases before any solve or conversion.  This script
# deliberately parses the source files as-is: it never relaxes or rewrites a
# case.  Its report is the provenance/compatibility gate for OOD evaluation.

using PowerModels
using JSON3
using SHA
using Dates

function _xfmr(b)
    haskey(b, "transformer") && return Bool(b["transformer"])
    abs(Float64(get(b, "tap", 1.0)) - 1.0) > 1e-8 ||
        abs(Float64(get(b, "shift", 0.0))) > 1e-8
end

function _ints(d)
    sort(parse.(Int, collect(keys(d))))
end

function _cost_audit(net)
    gens = values(get(net, "gen", Dict{String,Any}()))
    models = Dict{String,Int}()
    lengths = Dict{String,Int}()
    zero_cost = 0
    unsupported = Any[]
    for g in gens
        model = Int(get(g, "model", 2))
        cost = try Float64.(g["cost"]) catch; Float64[] end
        models[string(model)] = get(models, string(model), 0) + 1
        lengths[string(length(cost))] = get(lengths, string(length(cost)), 0) + 1
        isempty(cost) && (zero_cost += 1)
        if model != 2 || length(cost) > 3
            push!(unsupported, Dict(
                "generator_index" => Int(get(g, "index", -1)),
                "model" => model,
                "cost_length" => length(cost),
                "cost" => cost,
            ))
        end
    end
    Dict(
        "model_counts" => models,
        "coefficient_length_counts" => lengths,
        "zero_cost_generator_count" => zero_cost,
        "exact_quadratic_export_supported" => isempty(unsupported),
        "unsupported_generators" => unsupported,
    )
end

function audit(path)
    raw = read(path)
    net = PowerModels.parse_file(path; import_all=false, validate=true)
    branches = collect(values(get(net, "branch", Dict{String,Any}())))
    active_branches = filter(b -> Int(get(b, "br_status", 1)) == 1, branches)
    rates = [Float64(get(b, "rate_a", 0.0)) for b in active_branches]
    Dict(
        "source_path" => abspath(path),
        "source_sha256" => bytes2hex(sha256(raw)),
        "case_name" => get(net, "name", basename(path)),
        "baseMVA" => Float64(get(net, "baseMVA", NaN)),
        "bus_count" => length(get(net, "bus", Dict{String,Any}())),
        "generator_count" => length(get(net, "gen", Dict{String,Any}())),
        "active_generator_count" => count(g -> Int(get(g, "gen_status", get(g, "status", 1))) == 1,
                                          values(get(net, "gen", Dict{String,Any}()))),
        "load_count" => length(get(net, "load", Dict{String,Any}())),
        "shunt_count" => length(get(net, "shunt", Dict{String,Any}())),
        "branch_count" => length(branches),
        "active_branch_count" => length(active_branches),
        "ac_line_count" => count(b -> !_xfmr(b), active_branches),
        "transformer_count" => count(_xfmr, active_branches),
        "active_branches_with_positive_rate_a" => count(>(0.0), rates),
        "cost_audit" => _cost_audit(net),
    )
end

function main()
    length(ARGS) >= 2 || error("Usage: audit_pglib_inputs.jl <out.json> <case1.m> [case2.m ...]")
    out, paths = ARGS[1], ARGS[2:end]
    report = Dict(
        "purpose" => "Raw PGLib MATPOWER parse and exact-cost compatibility audit; no solve or relaxation performed.",
        "generated_at_utc" => string(now(UTC)),
        "cases" => [audit(p) for p in paths],
    )
    mkpath(dirname(out))
    open(out, "w") do io
        JSON3.pretty(io, report)
    end
    println("Wrote audit report: $out")
end

main()
