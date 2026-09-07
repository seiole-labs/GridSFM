module MixedPerturbations

using Random
using YAML

export apply_branch_derating!, apply_cost_shuffle!, apply_generator_outage!,
       apply_load_perturbation!, apply_mixed_perturbations,
       apply_voltage_tightening!, load_perturbation_config

const DEFAULT_CONFIG_PATH = normpath(joinpath(
    @__DIR__, "..", "..", "..", "benchmark", "perturbations.yaml"
))


function _require_range(section, lower_key, upper_key)
    lower = Float64(section[lower_key])
    upper = Float64(section[upper_key])
    lower <= upper || error("$lower_key must be <= $upper_key")
    return lower, upper
end


function _activation(section, rng::AbstractRNG)
    probability = Float64(section["activation_probability"])
    draw = rand(rng)
    return draw < probability, probability, draw
end


function _inactive_metadata(probability, draw; reason=nothing)
    metadata = Dict{String,Any}(
        "active" => false,
        "activation_probability" => probability,
        "activation_draw" => draw,
    )
    reason === nothing || (metadata["reason"] = reason)
    return metadata
end


function _weighted_sample_without_replacement(items, weights, count, rng::AbstractRNG)
    remaining_items = copy(items)
    remaining_weights = Float64.(weights)
    selected = eltype(items)[]
    for _ in 1:min(count, length(remaining_items))
        total = sum(remaining_weights)
        total > 0.0 || error("generator selection weights must sum to > 0")
        target = rand(rng) * total
        cumulative = 0.0
        chosen = lastindex(remaining_items)
        for index in eachindex(remaining_items)
            cumulative += remaining_weights[index]
            if target < cumulative
                chosen = index
                break
            end
        end
        push!(selected, remaining_items[chosen])
        deleteat!(remaining_items, chosen)
        deleteat!(remaining_weights, chosen)
    end
    return selected
end


function _sample_outage_count(section, rng::AbstractRNG)
    values = Int.(section["outage_count_values"])
    probabilities = Float64.(section["outage_count_probabilities"])
    length(values) == length(probabilities) ||
        error("outage count values and probabilities must have equal length")
    isapprox(sum(probabilities), 1.0; atol=1e-12) ||
        error("outage count probabilities must sum to 1")
    draw = rand(rng)
    cumulative = 0.0
    for (value, probability) in zip(values, probabilities)
        cumulative += probability
        draw < cumulative && return value
    end
    return values[end]
end


"""Load and minimally validate the benchmark perturbation contract."""
function load_perturbation_config(path::AbstractString=DEFAULT_CONFIG_PATH)
    config = YAML.load_file(path)
    Int(config["version"]) == 1 || error("unsupported perturbation config version")

    expected_order = [
        "load", "generator_outage", "branch_derating",
        "voltage_tightening", "cost_shuffle",
    ]
    String.(config["transform_order"]) == expected_order ||
        error("transform_order differs from the benchmark contract")

    perturbations = config["perturbations"]
    for name in expected_order
        haskey(perturbations, name) || error("missing perturbation section: $name")
        probability = Float64(perturbations[name]["activation_probability"])
        0.0 <= probability <= 1.0 ||
            error("$name activation_probability must be in [0, 1]")
    end

    load = perturbations["load"]
    _require_range(load, "system_factor_min", "system_factor_max")
    _require_range(load, "per_load_jitter_min", "per_load_jitter_max")

    outage = perturbations["generator_outage"]
    _sample_outage_count(outage, MersenneTwister(0))
    Int(outage["minimum_active_generators"]) >= 1 ||
        error("minimum_active_generators must be >= 1")

    derating = perturbations["branch_derating"]
    fraction = Float64(derating["selected_branch_fraction"])
    0.0 <= fraction <= 1.0 || error("selected_branch_fraction must be in [0, 1]")
    _require_range(derating, "rating_factor_min", "rating_factor_max")

    voltage = perturbations["voltage_tightening"]
    voltage_fraction = Float64(voltage["selected_bus_fraction"])
    0.0 <= voltage_fraction <= 1.0 ||
        error("selected_bus_fraction must be in [0, 1]")
    _require_range(voltage, "vmin_increase_min", "vmin_increase_max")
    _require_range(voltage, "vmax_decrease_min", "vmax_decrease_max")

    costs = perturbations["cost_shuffle"]
    cost_fraction = Float64(costs["selected_generator_fraction"])
    0.0 <= cost_fraction <= 1.0 ||
        error("selected_generator_fraction must be in [0, 1]")
    return config
end


"""
Apply the load portion of the mixed perturbation contract in place.

The caller owns `rng`; repeating with the same input and RNG seed is exact.
Returns compact metadata containing the values used and before/after totals.
"""
function apply_load_perturbation!(data, config, rng::AbstractRNG)
    load_config = config["perturbations"]["load"]
    sf_min, sf_max = _require_range(
        load_config, "system_factor_min", "system_factor_max"
    )
    jitter_min, jitter_max = _require_range(
        load_config, "per_load_jitter_min", "per_load_jitter_max"
    )
    shared_jitter = Bool(load_config["share_pd_qd_jitter"])
    system_factor = sf_min + rand(rng) * (sf_max - sf_min)

    loads = get(data, "load", Dict())
    total_pd_before = sum(Float64(get(load, "pd", 0.0)) for load in values(loads))
    total_qd_before = sum(Float64(get(load, "qd", 0.0)) for load in values(loads))

    for load_id in sort!(collect(keys(loads)); by=string)
        load = loads[load_id]
        pd_jitter = jitter_min + rand(rng) * (jitter_max - jitter_min)
        qd_jitter = shared_jitter ? pd_jitter :
            jitter_min + rand(rng) * (jitter_max - jitter_min)
        load["pd"] = Float64(get(load, "pd", 0.0)) * system_factor * pd_jitter
        load["qd"] = Float64(get(load, "qd", 0.0)) * system_factor * qd_jitter
    end

    total_pd_after = sum(Float64(get(load, "pd", 0.0)) for load in values(loads))
    total_qd_after = sum(Float64(get(load, "qd", 0.0)) for load in values(loads))
    return Dict{String,Any}(
        "active" => true,
        "system_factor" => system_factor,
        "per_load_jitter_min" => jitter_min,
        "per_load_jitter_max" => jitter_max,
        "share_pd_qd_jitter" => shared_jitter,
        "affected_loads" => length(loads),
        "total_pd_before" => total_pd_before,
        "total_pd_after" => total_pd_after,
        "total_qd_before" => total_qd_before,
        "total_qd_after" => total_qd_after,
    )
end


"""Probabilistically disable one or more active generators, weighted by Pmax."""
function apply_generator_outage!(data, config, rng::AbstractRNG)
    section = config["perturbations"]["generator_outage"]
    activated, probability, activation_draw = _activation(section, rng)
    activated || return _inactive_metadata(probability, activation_draw)

    minimum_pmax = Float64(section["minimum_pmax"])
    minimum_active = Int(section["minimum_active_generators"])
    generators = get(data, "gen", Dict())
    active = [
        (String(id), generator)
        for (id, generator) in generators
        if Int(get(generator, "gen_status", get(generator, "status", 1))) == 1
    ]
    candidates = sort!(
        [item for item in active if Float64(get(item[2], "pmax", 0.0)) > minimum_pmax];
        by=first,
    )
    maximum_outages = max(0, length(active) - minimum_active)
    maximum_outages > 0 && !isempty(candidates) ||
        return _inactive_metadata(probability, activation_draw; reason="insufficient_generators")

    requested_count = _sample_outage_count(section, rng)
    outage_count = min(requested_count, maximum_outages, length(candidates))
    candidate_ids = first.(candidates)
    weights = [Float64(get(generator, "pmax", 0.0)) for (_, generator) in candidates]
    selected_ids = _weighted_sample_without_replacement(
        candidate_ids, weights, outage_count, rng
    )

    removed_pmax = 0.0
    for generator_id in selected_ids
        generator = generators[generator_id]
        removed_pmax += Float64(get(generator, "pmax", 0.0))
        generator["gen_status"] = 0
    end
    return Dict{String,Any}(
        "active" => true,
        "activation_probability" => probability,
        "activation_draw" => activation_draw,
        "requested_outage_count" => requested_count,
        "outage_count" => outage_count,
        "generator_ids" => selected_ids,
        "removed_pmax" => removed_pmax,
        "selection_weight" => String(section["selection_weight"]),
    )
end


"""Probabilistically reduce rate_a/b/c on a fraction of active rated branches."""
function apply_branch_derating!(data, config, rng::AbstractRNG)
    section = config["perturbations"]["branch_derating"]
    activated, probability, activation_draw = _activation(section, rng)
    activated || return _inactive_metadata(probability, activation_draw)

    factor_min, factor_max = _require_range(
        section, "rating_factor_min", "rating_factor_max"
    )
    fraction = Float64(section["selected_branch_fraction"])
    branches = get(data, "branch", Dict())
    eligible_ids = sort!([
        String(id)
        for (id, branch) in branches
        if Int(get(branch, "br_status", 1)) == 1 &&
           Float64(get(branch, "rate_a", 0.0)) > 0.0
    ])
    isempty(eligible_ids) &&
        return _inactive_metadata(probability, activation_draw; reason="no_rated_branches")

    selected_count = max(1, round(Int, length(eligible_ids) * fraction))
    selected_ids = eligible_ids[randperm(rng, length(eligible_ids))[1:selected_count]]
    changes = Vector{Dict{String,Any}}()
    for branch_id in selected_ids
        branch = branches[branch_id]
        factor = factor_min + rand(rng) * (factor_max - factor_min)
        before = Dict{String,Float64}()
        after = Dict{String,Float64}()
        for rating in ("rate_a", "rate_b", "rate_c")
            if haskey(branch, rating)
                before[rating] = Float64(branch[rating])
                branch[rating] = before[rating] * factor
                after[rating] = Float64(branch[rating])
            end
        end
        push!(changes, Dict{String,Any}(
            "branch_id" => branch_id,
            "factor" => factor,
            "before" => before,
            "after" => after,
        ))
    end
    return Dict{String,Any}(
        "active" => true,
        "activation_probability" => probability,
        "activation_draw" => activation_draw,
        "eligible_branches" => length(eligible_ids),
        "selected_branch_fraction" => fraction,
        "selected_branches" => length(selected_ids),
        "changes" => changes,
    )
end


"""Probabilistically tighten voltage-magnitude bounds on selected buses."""
function apply_voltage_tightening!(data, config, rng::AbstractRNG)
    section = config["perturbations"]["voltage_tightening"]
    activated, probability, activation_draw = _activation(section, rng)
    activated || return _inactive_metadata(probability, activation_draw)

    vmin_delta_min, vmin_delta_max = _require_range(
        section, "vmin_increase_min", "vmin_increase_max"
    )
    vmax_delta_min, vmax_delta_max = _require_range(
        section, "vmax_decrease_min", "vmax_decrease_max"
    )
    fraction = Float64(section["selected_bus_fraction"])
    buses = get(data, "bus", Dict())
    bus_ids = sort!(String.(collect(keys(buses))))
    isempty(bus_ids) &&
        return _inactive_metadata(probability, activation_draw; reason="no_buses")

    selected_count = max(1, round(Int, length(bus_ids) * fraction))
    selected_ids = bus_ids[randperm(rng, length(bus_ids))[1:selected_count]]
    changes = Vector{Dict{String,Any}}()
    applied_count = 0
    for bus_id in selected_ids
        bus = buses[bus_id]
        vmin_before = Float64(get(bus, "vmin", 0.9))
        vmax_before = Float64(get(bus, "vmax", 1.1))
        vmin_delta = vmin_delta_min + rand(rng) * (vmin_delta_max - vmin_delta_min)
        vmax_delta = vmax_delta_min + rand(rng) * (vmax_delta_max - vmax_delta_min)
        vmin_after = vmin_before + vmin_delta
        vmax_after = vmax_before - vmax_delta
        applied = vmin_after < vmax_after
        if applied
            bus["vmin"] = vmin_after
            bus["vmax"] = vmax_after
            applied_count += 1
        else
            vmin_after = vmin_before
            vmax_after = vmax_before
        end
        push!(changes, Dict{String,Any}(
            "bus_id" => bus_id,
            "applied" => applied,
            "vmin_delta" => vmin_delta,
            "vmax_delta" => vmax_delta,
            "vmin_before" => vmin_before,
            "vmin_after" => vmin_after,
            "vmax_before" => vmax_before,
            "vmax_after" => vmax_after,
        ))
    end
    return Dict{String,Any}(
        "active" => applied_count > 0,
        "activation_probability" => probability,
        "activation_draw" => activation_draw,
        "selected_bus_fraction" => fraction,
        "selected_buses" => length(selected_ids),
        "tightened_buses" => applied_count,
        "changes" => changes,
    )
end


"""Shuffle complete cost arrays among compatible active generators."""
function apply_cost_shuffle!(data, config, rng::AbstractRNG)
    section = config["perturbations"]["cost_shuffle"]
    activated, probability, activation_draw = _activation(section, rng)
    activated || return _inactive_metadata(probability, activation_draw)

    generators = get(data, "gen", Dict())
    groups = Dict{Tuple{Int,Int},Vector{String}}()
    for generator_id in sort!(String.(collect(keys(generators))))
        generator = generators[generator_id]
        status = Int(get(generator, "gen_status", get(generator, "status", 1)))
        cost = get(generator, "cost", Any[])
        status == 1 && !isempty(cost) || continue
        compatibility_key = (
            Int(get(generator, "model", 2)),
            Int(get(generator, "ncost", length(cost))),
        )
        push!(get!(groups, compatibility_key, String[]), generator_id)
    end

    fraction = Float64(section["selected_generator_fraction"])
    mappings = Vector{Dict{String,Any}}()
    selected_total = 0
    for compatibility_key in sort!(collect(keys(groups)))
        group = groups[compatibility_key]
        length(group) >= 2 || continue
        selected_count = clamp(round(Int, length(group) * fraction), 2, length(group))
        selected_ids = group[randperm(rng, length(group))[1:selected_count]]
        original_costs = [deepcopy(generators[id]["cost"]) for id in selected_ids]
        # A non-zero cyclic shift guarantees every selected generator receives
        # another selected generator's complete cost curve.
        shift = rand(rng, 1:(selected_count - 1))
        source_ids = circshift(selected_ids, shift)
        source_costs = circshift(original_costs, shift)
        for (target_id, source_id, source_cost) in
            zip(selected_ids, source_ids, source_costs)
            generators[target_id]["cost"] = source_cost
            push!(mappings, Dict{String,Any}(
                "generator_id" => target_id,
                "cost_source_generator_id" => source_id,
                "cost" => deepcopy(source_cost),
            ))
        end
        selected_total += selected_count
    end

    selected_total > 0 ||
        return _inactive_metadata(probability, activation_draw; reason="no_compatible_cost_group")
    active_generator_count = count(generator ->
        Int(get(generator, "gen_status", get(generator, "status", 1))) == 1,
        values(generators),
    )
    eligible_cost_generator_count = sum(length, values(groups))
    return Dict{String,Any}(
        "active" => true,
        "activation_probability" => probability,
        "activation_draw" => activation_draw,
        "requested_generator_fraction" => fraction,
        "active_generators" => active_generator_count,
        "eligible_cost_generators" => eligible_cost_generator_count,
        "selected_generators" => selected_total,
        "actual_generator_fraction" =>
            selected_total / max(1, eligible_cost_generator_count),
        "mappings" => mappings,
    )
end


"""Apply the complete perturbation chain to a fresh deep copy of `base`."""
function apply_mixed_perturbations(base, config, master_seed::Integer)
    transformed = deepcopy(base)
    names = String.(config["transform_order"])
    functions = Dict(
        "load" => apply_load_perturbation!,
        "generator_outage" => apply_generator_outage!,
        "branch_derating" => apply_branch_derating!,
        "voltage_tightening" => apply_voltage_tightening!,
        "cost_shuffle" => apply_cost_shuffle!,
    )
    perturbations = Dict{String,Any}()
    sub_seeds = Dict{String,Int}()
    for (index, name) in enumerate(names)
        sub_seed = Int(master_seed) + 1_000_003 * index
        sub_seeds[name] = sub_seed
        perturbations[name] = functions[name](
            transformed, config, MersenneTwister(sub_seed)
        )
    end
    metadata = Dict{String,Any}(
        "seed" => Int(master_seed),
        "transform_order" => names,
        "sub_seeds" => sub_seeds,
        "perturbations" => perturbations,
    )
    return transformed, metadata
end

end # module
