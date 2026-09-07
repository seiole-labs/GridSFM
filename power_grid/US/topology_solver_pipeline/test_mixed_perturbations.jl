using Test
using Random

include(joinpath(@__DIR__, "mixed_perturbations.jl"))
using .MixedPerturbations


@testset "perturbation contract" begin
    config = load_perturbation_config()
    @test config["profile"] == "paper_id_mixed"
    @test config["transform_order"][1] == "load"
    @test config["perturbations"]["load"]["activation_probability"] == 1.0
end


@testset "generator outage" begin
    config = load_perturbation_config()
    section = config["perturbations"]["generator_outage"]
    section["activation_probability"] = 1.0
    section["outage_count_probabilities"] = [1.0, 0.0, 0.0]
    base = Dict{String,Any}(
        "gen" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "gen_status" => 1, "pmax" => Float64(id * 10), "pmin" => 0.0,
            ) for id in 1:4
        ),
    )
    transformed = deepcopy(base)

    metadata = apply_generator_outage!(
        transformed, config, MersenneTwister(42)
    )

    @test metadata["active"]
    @test metadata["outage_count"] == 1
    @test length(metadata["generator_ids"]) == 1
    @test count(generator -> generator["gen_status"] == 1,
                values(transformed["gen"])) == 3
    @test metadata["removed_pmax"] > 0.0
    @test base["gen"] != transformed["gen"]
    @test all(generator["gen_status"] == 1 for generator in values(base["gen"]))

    inactive_config = deepcopy(config)
    inactive_config["perturbations"]["generator_outage"]["activation_probability"] = 0.0
    unchanged = deepcopy(base)
    inactive = apply_generator_outage!(unchanged, inactive_config, MersenneTwister(42))
    @test !inactive["active"]
    @test unchanged == base
end


@testset "branch derating" begin
    config = load_perturbation_config()
    section = config["perturbations"]["branch_derating"]
    section["activation_probability"] = 1.0
    section["selected_branch_fraction"] = 0.20
    section["rating_factor_min"] = 0.80
    section["rating_factor_max"] = 0.80
    base = Dict{String,Any}(
        "branch" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "br_status" => 1, "rate_a" => 100.0,
                "rate_b" => 110.0, "rate_c" => 120.0,
                "br_r" => 0.01, "br_x" => 0.10,
            ) for id in 1:10
        ),
    )
    transformed = deepcopy(base)

    metadata = apply_branch_derating!(
        transformed, config, MersenneTwister(42)
    )

    @test metadata["active"]
    @test metadata["selected_branches"] == 2
    @test length(metadata["changes"]) == 2
    for change in metadata["changes"]
        branch = transformed["branch"][change["branch_id"]]
        @test change["factor"] == 0.80
        @test branch["rate_a"] == 80.0
        @test branch["rate_b"] == 88.0
        @test branch["rate_c"] == 96.0
        @test branch["br_r"] == 0.01
        @test branch["br_x"] == 0.10
    end
    @test all(branch["rate_a"] == 100.0 for branch in values(base["branch"]))
end


@testset "voltage tightening" begin
    config = load_perturbation_config()
    section = config["perturbations"]["voltage_tightening"]
    section["activation_probability"] = 1.0
    section["selected_bus_fraction"] = 0.20
    section["vmin_increase_min"] = 0.005
    section["vmin_increase_max"] = 0.005
    section["vmax_decrease_min"] = 0.005
    section["vmax_decrease_max"] = 0.005
    base = Dict{String,Any}(
        "bus" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "vmin" => 0.90, "vmax" => 1.10, "base_kv" => 230.0,
            ) for id in 1:10
        ),
    )
    transformed = deepcopy(base)

    metadata = apply_voltage_tightening!(
        transformed, config, MersenneTwister(42)
    )

    @test metadata["active"]
    @test metadata["selected_buses"] == 2
    @test metadata["tightened_buses"] == 2
    @test length(metadata["changes"]) == 2
    for change in metadata["changes"]
        @test change["applied"]
        @test change["vmin_after"] ≈ 0.905
        @test change["vmax_after"] ≈ 1.095
        bus = transformed["bus"][change["bus_id"]]
        @test bus["vmin"] == change["vmin_after"]
        @test bus["vmax"] == change["vmax_after"]
        @test bus["base_kv"] == 230.0
    end
    @test all(bus["vmin"] == 0.90 && bus["vmax"] == 1.10
              for bus in values(base["bus"]))

    crossing_config = deepcopy(config)
    crossing_config["perturbations"]["voltage_tightening"]["selected_bus_fraction"] = 1.0
    narrow = Dict{String,Any}(
        "bus" => Dict{String,Any}(
            "1" => Dict{String,Any}("vmin" => 0.999, "vmax" => 1.001),
        ),
    )
    crossing = apply_voltage_tightening!(
        narrow, crossing_config, MersenneTwister(42)
    )
    @test !crossing["active"]
    @test crossing["tightened_buses"] == 0
    @test narrow["bus"]["1"]["vmin"] == 0.999
    @test narrow["bus"]["1"]["vmax"] == 1.001
end


@testset "cost shuffle" begin
    config = load_perturbation_config()
    section = config["perturbations"]["cost_shuffle"]
    section["activation_probability"] = 1.0
    section["selected_generator_fraction"] = 0.50
    base = Dict{String,Any}(
        "gen" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "gen_status" => 1,
                "ncost" => 3,
                "cost" => [0.01 * id, 10.0 * id, 100.0 * id],
                "pmax" => 100.0 * id,
                "gen_bus" => id,
            ) for id in 1:4
        ),
    )
    transformed = deepcopy(base)
    costs_before = sort([copy(generator["cost"]) for generator in values(base["gen"])])

    metadata = apply_cost_shuffle!(transformed, config, MersenneTwister(42))

    @test metadata["active"]
    @test metadata["selected_generators"] == 2
    @test metadata["eligible_cost_generators"] == 4
    @test metadata["actual_generator_fraction"] == 0.5
    @test length(metadata["mappings"]) == 2
    @test all(mapping["generator_id"] != mapping["cost_source_generator_id"]
              for mapping in metadata["mappings"])
    costs_after = sort([copy(generator["cost"]) for generator in values(transformed["gen"])])
    @test costs_after == costs_before
    for (id, generator) in transformed["gen"]
        @test generator["pmax"] == base["gen"][id]["pmax"]
        @test generator["gen_bus"] == base["gen"][id]["gen_bus"]
    end
end


@testset "mixed orchestration" begin
    config = load_perturbation_config()
    base = Dict{String,Any}(
        "load" => Dict{String,Any}(
            string(id) => Dict{String,Any}("pd" => 1.0, "qd" => 0.4)
            for id in 1:4
        ),
        "gen" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "gen_status" => 1, "pmax" => 10.0 * id, "ncost" => 3,
                "cost" => [0.01 * id, 10.0 * id, 100.0 * id],
            ) for id in 1:4
        ),
        "branch" => Dict{String,Any}(
            string(id) => Dict{String,Any}(
                "br_status" => 1, "rate_a" => 100.0,
                "rate_b" => 110.0, "rate_c" => 120.0,
            ) for id in 1:10
        ),
        "bus" => Dict{String,Any}(
            string(id) => Dict{String,Any}("vmin" => 0.90, "vmax" => 1.10)
            for id in 1:10
        ),
    )

    first, first_metadata = apply_mixed_perturbations(base, config, 42_001)
    second, second_metadata = apply_mixed_perturbations(base, config, 42_001)

    @test first == second
    @test first_metadata == second_metadata
    @test base["load"]["1"]["pd"] == 1.0
    @test Set(keys(first_metadata["perturbations"])) == Set(config["transform_order"])
    @test length(unique(values(first_metadata["sub_seeds"]))) == 5

    # Changing load's random consumption cannot alter any later transform.
    shared_config = deepcopy(config)
    shared_config["perturbations"]["load"]["share_pd_qd_jitter"] = true
    _, shared_metadata = apply_mixed_perturbations(base, shared_config, 42_001)
    for name in config["transform_order"][2:end]
        @test first_metadata["perturbations"][name] ==
              shared_metadata["perturbations"][name]
    end
end


@testset "load perturbation" begin
    config = load_perturbation_config()
    base = Dict{String,Any}(
        "load" => Dict{String,Any}(
            "2" => Dict{String,Any}("pd" => 2.0, "qd" => 0.8),
            "1" => Dict{String,Any}("pd" => 1.0, "qd" => 0.4),
        ),
    )
    first = deepcopy(base)
    second = deepcopy(base)

    metadata_first = apply_load_perturbation!(first, config, MersenneTwister(42))
    metadata_second = apply_load_perturbation!(second, config, MersenneTwister(42))

    @test first == second
    @test metadata_first == metadata_second
    @test base["load"]["1"]["pd"] == 1.0
    @test metadata_first["affected_loads"] == 2
    @test 0.8 <= metadata_first["system_factor"] <= 1.5
    @test metadata_first["total_pd_before"] == 3.0
    @test metadata_first["total_qd_before"] ≈ 1.2
    @test metadata_first["total_pd_after"] > 0.0
    @test metadata_first["total_qd_after"] > 0.0

    for load in values(first["load"])
        @test load["pd"] > 0.0
        @test load["qd"] > 0.0
    end
end
