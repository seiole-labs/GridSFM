# GridSFM Pilot Perturbations

Perturbations modify the AC-OPF input problem. They do not directly modify its solution. After transformation, AC-OPF and GridSFM independently solve or predict against the same transformed grid.

The constants are defined in [`perturbations.yaml`](perturbations.yaml).

## Load variation

Changes active and reactive demand:

```text
new Pd = old Pd × system factor × local Pd jitter
new Qd = old Qd × system factor × local Qd jitter
```

- System factor: `0.8–1.5`, shared across the network.
- Local jitter: `0.9–1.1`, sampled per load.
- The current contract samples separate `Pd` and `Qd` jitter, which can change power factor slightly.
- Changed fields: `load.pd`, `load.qd`.
- Unchanged: topology, generator capacity, branch parameters, voltage limits, and costs.

The two multipliers have different roles. The system factor creates a correlated network-wide operating condition, such as a light-load or peak-load hour. Local jitter creates spatial variation around that condition. For a grid with many loads, positive and negative local jitters tend to average toward `1.0`, leaving the system factor as the main control of total demand. An individual load can remain near its original value—for example, `0.9 × 1.1 = 0.99`—and that is intentional.

Example:

```text
Original: Pd=100 MW, Qd=30 MVAr
System factor=1.20, Pd jitter=0.95, Qd jitter=1.04
Result: Pd=114 MW, Qd=37.44 MVAr
```

## Generator outage

Marks selected generators unavailable:

```text
gen.gen_status: 1 → 0
```

- Activates with probability `0.30`.
- Removes 1/2/3 generators with probabilities `0.70/0.20/0.10`.
- Keeps at least two generators active.
- Selection is planned to be weighted by generator `Pmax`.
- Changed field: `gen.gen_status`.
- Unchanged: loads, branch parameters, and voltage limits.

The remaining generators must redispatch. This can increase congestion or make the problem infeasible.

The probabilities are hierarchical. A scenario first has a 30% chance of activating the outage transformation. Conditional on activation, it removes exactly 1, 2, or 3 generators with probabilities 70%, 20%, and 10%. Therefore the unconditional scenario probabilities are 21% for exactly one outage, 6% for exactly two, 3% for exactly three, and 70% for no generator outage. These probabilities describe the number of outages, not particular generator identities. Generator identities are selected afterward from the eligible fleet.

## Branch derating

Reduces selected branch operating capacities:

```text
new rating = old rating × derating factor
```

- Activates with probability `0.20`.
- Selects approximately 10% of active rated branches.
- Rating factor: `0.70–0.95`.
- Changed fields: `branch.rate_a`, `branch.rate_b`, `branch.rate_c`.
- Unchanged: resistance, reactance, and connectivity.

A derated branch remains connected; this is not a line outage.

## Voltage-limit tightening

Narrows the allowed voltage-magnitude band at selected buses:

```text
new Vmin = old Vmin + sampled increase
new Vmax = old Vmax - sampled decrease
```

- Activates with probability `0.15`.
- Selects approximately 10% of buses.
- Each bound moves inward by `0.00–0.01 pu`.
- Changed fields: `bus.vmin`, `bus.vmax`.
- The transformation must never allow `Vmin >= Vmax`.

This can increase reactive-power stress and make voltage feasibility harder.

## Generator-cost reshuffling

Changes the economic merit order by permuting complete cost curves among compatible active generators.

- Activates with probability `1.0` in the current proposed contract.
- Selects approximately 40% of active generators.
- Only exchanges compatible cost-function shapes.
- Changed field: `gen.cost`.
- Unchanged: generator capacity, reactive limits, location, and status.

No new costs are invented. Existing cost curves are reassigned so the model must respond to the supplied objective rather than memorize a topology's usual dispatch.

### Worked cost-shuffle example

Assume three active generators have quadratic production costs:

```text
Generator A: cost(Pg) = 0.01 Pg² + 10 Pg + 100
Generator B: cost(Pg) = 0.02 Pg² + 20 Pg + 200
Generator C: cost(Pg) = 0.03 Pg² + 30 Pg + 300
```

Their cost arrays are:

```text
A: [0.01, 10, 100]
B: [0.02, 20, 200]
C: [0.03, 30, 300]
```

Suppose A and C are selected and their complete cost curves are exchanged:

```text
After shuffling:
A: [0.03, 30, 300]  # previously C's cost
B: [0.02, 20, 200]  # unchanged; not selected
C: [0.01, 10, 100]  # previously A's cost
```

Only the cost arrays move. Generator A still has A's original bus, `Pmin/Pmax`, `Qmin/Qmax`, voltage setpoint, and status; the same is true for C. The shuffle therefore makes C economically preferable to A without moving or resizing either physical generator. AC-OPF recomputes dispatch using this new objective.

The transformation groups generators by compatible cost representation before shuffling. A quadratic three-coefficient curve is not exchanged with an incompatible piecewise-linear or differently shaped cost definition.

## Mixed scenario

Transformations are visited in this fixed order:

```text
load
→ generator outage
→ branch derating
→ voltage tightening
→ cost reshuffling
```

Each transformation has an independent deterministic random stream. Optional transformations record `active: false` when skipped. Every scenario begins from a fresh copy of the base topology; scenarios are never perturbed on top of one another.

All five transformations and the deterministic mixed-chain orchestration are implemented and tested. Scenario persistence is implemented by `generate_mixed_scenario.jl`.
