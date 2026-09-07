#!/usr/bin/env julia

using JSON3

include(joinpath(@__DIR__, "mixed_perturbations.jl"))
using .MixedPerturbations

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))


function repo_relative(path::AbstractString)
    return relpath(abspath(path), REPO_ROOT)
end


function main()
    selection_path = length(ARGS) >= 1 ? abspath(ARGS[1]) :
        joinpath(REPO_ROOT, "artifacts", "pilot", "selection.json")
    scenario_index = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    output_dir = length(ARGS) >= 3 ? abspath(ARGS[3]) :
        joinpath(REPO_ROOT, "artifacts", "pilot")
    topology_index = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    scenario_index >= 1 || error("scenario index must be >= 1")

    selection = JSON3.read(read(selection_path, String), Dict{String,Any})
    selected = selection["selected"]
    1 <= topology_index <= length(selected) || error("topology index is out of range")
    selected_topology = selected[topology_index]
    base_relative = String(selected_topology["path"])
    base_path = joinpath(REPO_ROOT, base_relative)
    base = JSON3.read(read(base_path, String), Dict{String,Any})
    config = load_perturbation_config()

    master_seed = Int(selection["seed"]) * 1000 + (topology_index - 1) * 100 + scenario_index
    sample_id = "$(selected_topology["name"])_mixed_$(lpad(scenario_index, 4, '0'))"
    transformed, perturbation_metadata = apply_mixed_perturbations(
        base, config, master_seed
    )

    mkpath(output_dir)
    transformed_path = joinpath(output_dir, "$(sample_id)_transformed.json")
    metadata_path = joinpath(output_dir, "$(sample_id)_metadata.json")
    open(transformed_path, "w") do io
        JSON3.pretty(io, transformed)
        write(io, '\n')
    end

    metadata = Dict{String,Any}(
        "sample_id" => sample_id,
        "base_grid" => repo_relative(base_path),
        "transformed_source" => repo_relative(transformed_path),
        "transformed_grid" => repo_relative(
            joinpath(output_dir, "$(sample_id).pyg.json")
        ),
        "seed" => master_seed,
        "perturbations" => perturbation_metadata["perturbations"],
    )
    open(metadata_path, "w") do io
        JSON3.pretty(io, metadata)
        write(io, '\n')
    end
    println("wrote $(repo_relative(transformed_path))")
    println("wrote $(repo_relative(metadata_path))")
end


main()
