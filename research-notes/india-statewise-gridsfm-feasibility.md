# Feasibility of an India state/UT-wise GridSFM dataset

**Research date:** 2026-09-09  
**Paper assessed:** Britto et al., *Building Power Grid Models from Open Data: A Complete Pipeline from OpenStreetMap to Optimal Power Flow*, arXiv:2605.04289v1 (May 2026), [HTML](https://arxiv.org/html/2605.04289v1).

## Decision

The paper's approach is **technically feasible for India as a research-grade, geographically grounded reconstructed transmission model**, but it is not a direct data-port. The topology can still come from OpenStreetMap (OSM); Survey of India (SoI) can provide authoritative current state/UT boundaries; and Central Electricity Authority (CEA), Census of India, and load-despatch data can calibrate aggregate generation and demand. The output must be described as reconstructed/synthetic, not as India's operational grid.

The recommended architecture is **one national master network, plus state/UT views that preserve interstate tie-lines and boundary equivalents**. Independently clipping and solving 36 political units would discard the very inter-state transfers on which many states and small Union Territories depend. State/UT is a useful dataset partition and reporting dimension, but not generally an electrical island.

Proceed with a gated pilot, not a full 36-unit build. First audit OSM completeness and produce one 500+ bus state plus its surrounding regional model. Continue only if topology, capacity, generation, and strict AC-OPF validation gates pass.

## What the paper requires

The paper implements five stages:

1. Extract OSM power lines, cables, substations, plants, converters, and tags through a local Overpass instance.
2. Reconstruct a bus-branch graph: infer missing voltage, parse circuits, merge fragmented ways, form voltage-specific buses, infer transformers, attach generators, detect HVDC, and retain a connected component.
3. Estimate line/transformer impedance, thermal ratings, generator limits, reactive capability, and quadratic costs.
4. Allocate hourly balancing-area demand to buses using population, initialize merit-order dispatch, and add missing official generators until adequate reserve exists.
5. Run DC- then AC-OPF in PowerModels/Ipopt, with a documented progressive relaxation ladder and synthetic shunts where missing reactive data prevents convergence.

The method is explicitly designed to create a plausible, solvable research network rather than recover a true utility model. Its principal empirical warnings transfer directly to India: OSM omits or inconsistently tags voltage and parallel circuits; electrical parameters are lookup-table estimates; population is only a load proxy; and convergence after large relaxations is a weak indication of physical fidelity ([paper Sections 3 and 5](https://arxiv.org/html/2605.04289v1#S3)).

## India source mapping

| Required input | Best India source | Availability and resolution | Fitness for this pipeline |
|---|---|---|---|
| Transmission geometry, substations, plants, converters | OSM `power=*` objects | Routable geometry and the same tags used by the paper. The OSM India power-network page reports mapped 765/400/220/110 kV infrastructure and substantial line/plant coverage, but also leaves mapping-quality work explicitly unfinished ([OSM India power-network page](https://wiki.openstreetmap.org/wiki/Power_networks/India)). | **Usable after a measured audit.** It is the only identified open nationwide source with line-level routable geometry. Do not treat the community statistics as an accuracy guarantee. |
| Current state/UT boundaries | Survey of India Administrative Boundary Database (ABDB) | Whole-country state-to-district and state-to-taluk Shapefiles are listed at 1:1 million and price ₹0 ([SoI digital products](https://onlinemaps.surveyofindia.gov.in/Digital_Products.aspx)); the 2025 catalogue also lists state/UT through village layers in SHP/GDB and ORGI harmonisation ([SoI vector catalogue](https://www.surveyofindia.gov.in/UserFiles/files/Vector%20Data%20Catalog%202025%281%29.pdf)). | **Authoritative clipping/assignment source.** Preserve product version and terms. A free price is not, by itself, an unrestricted licence. |
| Administrative identifiers | Ministry of Panchayati Raj LGD; ORGI Census codes | LGD provides current directory downloads and change records ([LGD downloads](https://lgdirectory.gov.in/demo/downloadDirectory.do)). The Census 2011 Location Code Directory provides state/UT, district, subdistrict, village and town codes ([ORGI directory](https://censusindia.gov.in/nada/index.php/catalog/42648)). | **Usable with a versioned concordance.** Census geography is historical; never join it to current SoI names alone. |
| Population for spatial load allocation | Census of India Primary Census Abstract (PCA) | The Population Finder exposes 85 indicators at district, subdistrict, town, village, and ward levels and CSV downloads ([ORGI Population Finder](https://censusindia.gov.in/census.website/en/data/population-finder)). The nationwide village-level PCA is downloadable ([ORGI catalogue](https://www.censusindia.gov.in/nada/index.php/catalog/42559)). | **Usable but stale.** Census 2011 predates major urban, industrial, and administrative changes. It is a baseline proxy, not a 2026 load map. |
| State demand and energy | CEA API/reports; SLDC/RLDC/NLDC feeds where reproducibly accessible | CEA exposes state-wise installed capacity, power-supply energy/peak, transformation capacity, transmission lines, generation, renewable energy, and consumption datasets ([CEA API catalogue](https://cea.nic.in/api-for-central-electricity-authority-data/?lang=en)); CEA publishes state-wise power-supply reports ([CEA power supply](https://cea.nic.in/power-supply/?lang=en)). The Indian Electricity Grid Code assigns active/reactive demand estimation to SLDCs and regional/national aggregation to RLDCs/NLDC ([CERC IEGC 2023](https://www.cercind.gov.in/Regulations/180-Regulations.pdf)). | **Good for totals and validation; incomplete as one uniform EIA-930 replacement.** Build source-specific adapters and archive raw snapshots. Do not silently mix annual peak, monthly energy, schedule, and actual-demand concepts. |
| Generator inventory/capacity/fuel | CEA station list and monthly installed-capacity reports, reconciled with OSM | CEA's Power Data Management Division publishes a station list, monthly installed capacity, state-wise capacity, generation, transmission and consumption statistics ([CEA PDM Division](https://cea.nic.in/power-data-management-division/?lang=en)). | **Adequate for aggregate reconciliation; uncertain for automatic plant-level geocoding.** Match names/capacity/fuel to OSM conservatively and record confidence. PPA allocation is not the same as physical plant location. |
| Transmission totals and planning benchmarks | CEA General Review, transmission reports, National Electricity Plan (NEP) | CEA publishes 220 kV-and-above line/substation completion reports ([CEA transmission reports](https://cea.nic.in/transmission-reports/?lang=en)); NEP Volume II provides the national transmission-planning baseline and state-wise generation assumptions ([NEP Volume II](https://cea.nic.in/wp-content/uploads/notification/2024/10/National_Electricity_Plan_Volume_II_Transmission.pdf)). | **Useful for voltage-class/state/region calibration and validation.** It does not replace line geometry or utility branch impedances. |
| India-specific electrical lookup tables | CEA planning/equipment standards | CEA's 2023 planning manual represents 765/400/230/220/132/110/66/33 kV bus levels ([Manual on Transmission Planning Criteria 2023](https://cea.nic.in/wp-content/uploads/document_upload/2023/06/Manual_on_Transmission_Planning_Criteria_2023.pdf)); CEA also publishes transformer/reactor specifications for 66 kV and above ([CEA engineering publications](https://cea.nic.in/power-system-engineering-technology-development-division/?lang=en)). | **Required adaptation.** Do not reuse US conductor, transformer, short-term-rating, or voltage-class tables without validation. |
| Terrain/land use (optional QA/enrichment) | SoI terrain products; NRSC Bhuvan | SoI lists countrywide 10 m DTM and map/vector products ([SoI portal description](https://onlinemaps.surveyofindia.gov.in/AboutPortal.aspx)). Bhuvan lists multiple LULC resolutions and delivery modes ([NRSC Bhuvan store](https://bhuvan-app1.nrsc.gov.in/2dresources/bhuvanstore.php)). | **Not needed for v1 OPF.** Useful later for route/hazard QA. Bhuvan's default terms restrict copying, derivatives, redistribution, and bulk download without permission ([Bhuvan terms](https://bhuvan.nrsc.gov.in/terms.php)). |

### No complete Indian equivalents were identified

The official source review did **not** identify an open, nationwide, machine-readable utility network containing line/substation geometry, branch impedances and ratings, transformer taps, reactive devices, and nodal hourly demand. It also did not identify a single open plant-level equivalent to the paper's combined EIA-860/EIA-923 stack with uniform location, capacity, operating status, heat rate, and fuel cost. These gaps do not prevent a reconstructed model; they prevent claims of operational accuracy and make calibration/uncertainty metadata essential.

### Initial OSM feasibility signal: Karnataka

A direct Overpass audit against the live OSM database on 2026-09-09 found 2,858 Karnataka `power=line`/`power=cable` ways, of which 2,741 (95.9%) had a `voltage` tag; 2,728 ways included at least one common Indian transmission voltage (66/110/132/220/400/765 kV). The same audit returned 1,824 `power=substation` features and 509 `power=plant` features. This is a strong signal that an OSM-based pilot is worth attempting, and Karnataka is a reasonable first candidate subject to boundary/interface checks.

These are raw OSM-feature counts, not electrical-model counts: one circuit can contain many ways, one substation can have several features and voltage buses, a plant feature need not contain validated capacity, and high tag coverage does not prove positional, circuit-count, connectivity, or completeness accuracy. The extracted data remains subject to ODbL. Freeze the exact OSM snapshot and query in Phase 0 so this live-data observation becomes reproducible.

## Required method changes for India

### 1. Build nationally, publish state/UT views

Use a national OSM extract and assign facilities/lines to SoI state/UT polygons. Retain every cross-boundary circuit once in the national graph. Derive a state view with:

- internal buses and branches;
- explicit interface buses and tie branches crossing its boundary;
- measured or scenario-dependent net interchange when obtainable, otherwise a labelled boundary equivalent;
- a manifest identifying parent national graph, cut set, source dates, and assumptions.

Also publish five region-level cases aligned with Indian grid-operation practice before a single all-India case. The IEGC applies across users, SLDCs, RLDCs, NLDC, CTU and state utilities, confirming that administrative and control scopes are nested rather than isolated ([CERC IEGC 2023](https://www.cercind.gov.in/Regulations/180-Regulations.pdf)).

### 2. Replace the US voltage rules and lookup tables

The paper's fixed `>=69 kV` filter would incorrectly delete India's 66 kV sub-transmission assets. Make the cutoff configurable, initially `>=66 kV`, and retain explicit voltage classes 66, 110, 132, 220/230, 400 and 765 kV. Treat 220 and 230 kV as close nominal classes for topology inference, not automatically as a transformer transition. Parameter tables must reflect Indian conductor bundles, voltage levels, transformer practices, 50 Hz operation, and CEA ratings. Every imputed value should carry method/version/confidence fields.

### 3. Recalibrate topology factors instead of copying US values

The paper's topology/capacity factors are calibrated against US route-mile/circuit-mile patterns and even change by another `x3`/`x2` for regional models. Those numbers have no evidentiary basis in India. Estimate India factors by voltage and, if the data supports it, by state/region:

1. deduplicate OSM corridors nationally before any state aggregation;
2. compare OSM geodesic route length and parsed circuit length with CEA circuit-km totals on a common date/status basis;
3. separately fit impedance-equivalent circuit count and thermal-capacity correction;
4. validate on held-out regions rather than tuning solely for OPF convergence.

Convergence alone is not calibration: the paper's progressive relaxation and dense synthetic shunts can make a materially wrong network solvable.

### 4. Make demand allocation hierarchical and uncertainty-aware

Use CEA/SLDC state totals for each timestamp or reporting period, then allocate to buses with a hierarchy such as district/town/village population, urban/commercial indicators, and known bulk/industrial loads where legally reusable. Retain a population-only baseline exactly analogous to the paper for reproducibility, but publish alternative allocations because 2011 population is old and industrial/agricultural electricity demand need not track population. Record whether each scenario uses actual demand, schedule, peak, energy-derived average, or synthetic scaling.

### 5. Handle generation and costs as separate confidence layers

Reconcile OSM plants against the CEA station list using name, technology, capacity, state and proximity; never overwrite a weak match as if confirmed. Keep physical installed capacity, contractual/PPA allocation, availability, dispatch, and economic cost distinct. If no lawful uniform plant heat-rate/fuel-price source is available, use transparent technology-level quadratic costs only to create an ordering for OPF. Do not interpret resulting marginal costs as Indian market prices.

### 6. Preserve HVDC explicitly

India's long-distance HVDC links and back-to-back stations cannot safely be collapsed into ordinary AC branches. The paper exports them as controllable `dcline` elements. The current repository's scenario exporter enumerates only PowerModels `branch` objects and emits only `ac_line` and `transformer` edge families; it has no `dcline` export path ([exporter](../power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl)). Until a GridSFM-compatible DC representation is designed and evaluated, either:

- keep HVDC in the PowerModels solve but declare that GridSFM sees an incomplete graph, which is unsuitable for faithful end-to-end inference; or
- create explicit converter/DC edge types and retrain/fine-tune the model.

Dropping HVDC silently is not acceptable.

## State/UT edge cases

| Case | Required treatment |
|---|---|
| Cross-border supply and transit | Never delete a line merely because it crosses a state polygon. State models need tie branches/boundary equivalents and net-interchange validation. DVC and direct ISTS entities also make simple state ownership/allocation misleading. |
| Andaman & Nicobar Islands and Lakshadweep | CEA describes them as stand-alone systems excluded from regional totals ([CEA September 2025 executive summary](https://cea.nic.in/wp-content/uploads/executive/2025/10/Executive_Summary_September_2025_Actual.pdf)). Model island components separately, with one reference/slack per connected AC island; do not force-connect them to mainland India. |
| Puducherry | It comprises separated enclaves supplied through neighbouring systems and central allocations ([Puducherry Electricity Department](https://electricity.py.gov.in/power-system-availability)). The state key must support MultiPolygon geography and multiple electrical components/interfaces. |
| Dadra and Nagar Haveli and Daman and Diu | The merged UT has three districts and separated territories ([UT administration](https://ddd.gov.in/introduction/)). Use the current unit name/code, MultiPolygon geometry, and a concordance to older records. |
| Jammu & Kashmir and Ladakh | Census 2011 geography predates the current two-UT arrangement. Maintain a dated code/boundary crosswalk. SoI's 2026 political map notes that J&K UT administrative boundaries are yet to be authenticated ([SoI Political Map, 13th ed.](https://surveyofindia.gov.in/UserFiles/files/POL_MAP_4M_ENGLISH_13thEdn2026%281%29.pdf)). |
| Arunachal Pradesh/Assam/Meghalaya boundaries | The same official SoI map says these interstate boundaries have to be verified. Preserve the official source geometry and caveat; do not “repair” it from an unversioned third-party map. |
| Delhi, Chandigarh, other small mainland UTs; Goa and small northeastern states | Likely below, or close to, the model's supported distribution after a 66 kV filter. Prefer regional inference and state-level reporting. The local model documentation warns that grids below 500 buses are outside the released checkpoint's training support ([model README](../model/README.md)). |
| Disconnected OSM components | Do not automatically keep only the largest component before diagnosing mapping gaps. Record retained/dropped bus, generation and demand fractions; legitimate islands and enclave components require separate slacks rather than deletion. |

Geometries should be exchanged in WGS84, but distance buffers, snapping and clustering must use geodesic distance or a suitable projected CRS. India crosses multiple UTM zones, so a single nationwide UTM projection is inappropriate for the national graph.

## Licensing and release feasibility

- OSM data is available under ODbL 1.0, with attribution and share-alike obligations for publicly used derivative databases ([OSM copyright](https://www.openstreetmap.org/copyright), [ODbL text](https://wiki.openstreetmap.org/wiki/Open_Database_License/ODbL-1.0.txt)). Before releasing combined PowerModels/GridSFM data, obtain a specific legal determination of whether it is a Derivative Database or Produced Work and package the required notices/source offer accordingly.
- Datasets actually published under the Government Open Data License–India permit worldwide royalty-free commercial and non-commercial reuse, adaptation, and derivative publication subject to attribution and related conditions ([GODL-India](https://data.gov.in/sites/default/files/NDSAP_OpenDataLicense.pdf)). Individual resource metadata still governs; linked datasets are not automatically GODL ([data.gov.in terms](https://www.data.gov.in/terms-of-use)).
- The 2021 Indian geospatial guidelines generally liberalise collection and use, reserve finer-than-threshold data creation/ownership and domestic storage/processing requirements in specified cases to Indian entities, and require SoI boundaries as the standard for political maps ([DST Guidelines](https://dst.gov.in/sites/default/files/Final%20Approved%20Guidelines%20on%20Geospatial%20Data_0.pdf)). Check the current negative lists before public release ([DST notice hub](https://dst.gov.in/node/5434)).
- SoI portal terms require attribution/compliance and constrain sharing downloaded data with foreign persons/entities except as permitted ([SoI cart/terms page](https://onlinemaps.surveyofindia.gov.in/addtocart.aspx)). Keep SoI geometry as a separately licensed input or clipping/QA dependency until redistribution rights are confirmed.
- Bhuvan's default terms are too restrictive for assuming open redistribution. Exclude Bhuvan-derived layers from the public v1 artifact unless the specific dataset grant expressly permits it.

The safest v1 release is therefore an OSM-derived electrical database with clear ODbL handling, reproducible scripts, aggregate CEA validation tables whose individual licences permit reuse, and instructions for users to obtain restricted SoI/Bhuvan inputs separately where necessary.

## Fit with this repository

The repository already contains reusable downstream pieces but is missing the central India build stage:

- The checked-in pipeline is explicitly **stage 2 onward**. It expects a PowerModels-compatible raw topology JSON; OSM extraction, multi-source joins, topology reconstruction and India LUT generation are not present ([pipeline details](../power_grid/US/topology_solver_pipeline/PIPELINE_DETAILS.md)). A new India topology-builder component is required.
- Once a valid raw PowerModels JSON exists, the Julia solver can progressively repair it to a cold-strict-solvable base, export `.pyg.json`, create `1 + 5N` solved perturbations per grid, and round-trip verify them ([pipeline README](../power_grid/US/topology_solver_pipeline/README.md)). This is a strong reuse point.
- GridSFM inference accepts the native `.pyg.json` schema and predicts bus voltage/angle, generator P/Q, branch flows, and feasibility. The schema has bus, generator, load, shunt, AC-line and transformer features ([model README](../model/README.md)).
- The released checkpoint is documented for grids of at least 500 buses. Small state/UT cases should not be presented as meaningful zero-shot results.
- Current fine-tuning supports OPFData-backed datasets only; arbitrary custom `.pyg.json` scenarios are inference-compatible but not supported by the fine-tuning path ([model README](../model/README.md#fine-tuning-v11-only-opfdata-only)). An India training project therefore needs an adapter/dataset path in addition to data creation.
- The current exporter omits HVDC, as noted above. This is an implementation blocker for a faithful India-wide GridSFM graph, though not for an AC-only pilot whose omitted interfaces are explicitly bounded and disclosed.

## Computational feasibility

The computational scale is manageable on research hardware; data quality and solver-label generation are the larger risks.

- In the paper, typical US states contain 100–4,000 buses and 200–6,500 branches. Its 5,076-bus Western case solved AC-OPF in about one minute at strict settings, while the 21,697-bus Eastern case required L3 and 47 minutes. These are reported results on the authors' system, not India runtime promises ([paper Results](https://arxiv.org/html/2605.04289v1#S4)).
- India state/region topology size is unknown until extraction. Geometry parsing, spatial indexing, snapping and union-find should be near-linear and are batchable by region.
- AC-OPF scenario labeling dominates. With the existing `1 + 5N` generator, `N=100` yields 501 solves per topology; three pilot topologies yield 1,503 scenarios. Scale only after measuring strict-solve success and wall time.
- Scenario generation already parallelises across Julia workers and resumes by skipping existing files. Cache and storage sizes should be measured from the pilot, not extrapolated from OSM feature counts.
- National or regional zero-shot GridSFM inference is computationally plausible, but validity is uncertain because India is out-of-distribution in geography, parameter assumptions, topology and operating patterns. AC-OPF ground truth and feasibility metrics remain mandatory.

## Validation gates

A state/region advances only if its versioned build report passes declared thresholds. Suggested initial gates are:

1. **Source completeness:** proportions of retained line length with valid voltage/circuit/frequency; substations with polygons or high-confidence points; generators matched to CEA; unresolved/dropped features.
2. **Aggregate reconciliation:** OSM-derived circuit-km, transformer capacity, physical installed generation by fuel, state demand/energy and regional totals versus same-date CEA values. Report ratios, never conceal scaling factors.
3. **Topology:** connected-component distribution; fraction of load and generation in each component; cross-border tie count/capacity; degree and radiality diagnostics; no accidental duplicate border corridors.
4. **Electrical sanity:** India-specific R/X/B and MVA ranges by voltage; transformer ratios/taps; 50 Hz treatment; no zero/negative impedances; DC/AC loss ranges; voltage and angle violations.
5. **Solver quality:** strict L0 first, achieved relaxation level, synthetic-shunt fraction/MVAr, load shedding, solve status/time, KCL residuals, thermal violations, and round-trip objective agreement. L4/L5 or material shedding fails physical-use qualification even if the solver returns success.
6. **GridSFM evaluation:** compare against AC-OPF labels by state/region and topology-held-out splits; report MAE, cost error, KCL, thermal overload, feasibility classification, and solver-warm-start reliability. Keep state families out of training to measure geographic/topology generalisation.

## Main risks and mitigations

| Risk | Severity | Mitigation |
|---|---:|---|
| OSM topology/tags are incomplete or regionally uneven | High | Run a nationwide tag/completeness audit first; publish confidence and dropped-feature metrics; calibrate against CEA aggregates; contribute corrections only from OSM-compatible sources. |
| State clipping creates fictitious islands or removes transit paths | High | National master graph; buffer extraction; stable OSM-ID deduplication; boundary buses and equivalents; region cases before isolated state cases. |
| US electrical LUT/scaling assumptions bias India | High | Replace with CEA/Indian voltage-class tables and separately fitted topology/capacity factors; validate out of sample. |
| No uniform plant-level heat-rate/cost dataset | Medium-high | Use costs as transparent scenario assumptions, not market estimates; sensitivity tests and confidence fields. |
| Census 2011 population misallocates modern/industrial load | High | Maintain population-only baseline plus alternate spatial allocations; validate district/city/industrial totals where available. |
| HVDC is lost between PowerModels and GridSFM | High | Add a represented DC/converter path before national claims; never silently drop `dcline`. |
| Small states/UTs are below 500 buses | High for zero-shot SFM | Infer on electrically meaningful region/national graphs and aggregate results to state; do not pad or invent buses merely to satisfy model size. |
| Relaxation/synthetic shunts hide bad inputs | High | Treat relaxation and shunt density as quality metrics; fail high-relaxation cases; do not tune only for convergence. |
| Licence/security restrictions block a redistributable bundle | High | Dataset-by-dataset licence ledger, source/date hashes, separated restricted layers, ODbL/GODL attribution, Indian legal/security review before release. |

## Phased implementation recommendation

### Phase 0 — evidence and licence gate (2–4 weeks)

- Freeze an OSM snapshot; obtain/version SoI boundaries and LGD/ORGI concordances.
- Audit India-wide power tags by state/UT and voltage: feature counts, route length, missing voltage/circuits/frequency, endpoint-to-substation coverage, generator capacity and cross-border duplicates.
- Build a licence/provenance manifest for every source and confirm the public-release design.
- Select one pilot state only after the audit. It should have at least 500 reconstructed buses, good high-voltage coverage, meaningful generation diversity, and tractable neighbouring interfaces.

**Go/no-go:** no full implementation if a suitable pilot cannot retain at least 500 buses with defensible voltage/connectivity data or if a redistributable source stack cannot be established.

### Phase 1 — AC-only topology pilot (4–8 weeks)

- Implement the missing OSM-to-PowerModels stage under a new India-specific package; keep raw, normalized and derived layers separate.
- Use SoI boundaries for assignment, a configurable 66 kV filter, geodesic/projected metric operations, national OSM-ID deduplication, and explicit interface buses.
- Build CEA-based India LUTs and confidence metadata.
- Run the existing solve/export/verify pipeline on the pilot state and its enclosing region; do not yet claim GridSFM accuracy.

**Go/no-go:** strict or only light-relaxation AC-OPF, no load shedding, bounded synthetic reactive compensation, plausible aggregate reconciliation, and successful round-trip verification.

### Phase 2 — demand, generation and scenarios (4–8 weeks)

- Add timestamped CEA/SLDC demand adapters, Census/LGD/SoI concordance, population baseline plus alternative load allocations, and conservative CEA-to-OSM plant matching.
- Generate a small scenario set (`N=100` per perturbation mode is enough for pipeline and distribution checks), including peak/off-peak and seasonal cases.
- Add India-relevant perturbations later: renewable availability, monsoon/heatwave load, interstate transfer stress, generator/line outage and HVDC availability. Keep every assumed profile explicit.

### Phase 3 — GridSFM benchmark and adaptation (4–6 weeks)

- Establish zero-shot results only on 500+ bus state/region graphs.
- Implement custom-India training support rather than assuming the current OPFData-only fine-tuner accepts pipeline `.pyg.json`.
- Use topology-held-out state/region splits and compare zero-shot, fine-tuned and AC-OPF baselines.
- Treat GridSFM as a surrogate proposal; verify feasibility and, for operationally meaningful studies, pass predictions through an AC solver.

### Phase 4 — regional/national scale and release

- Add explicit HVDC/converter representation and test it end-to-end.
- Build five regional cases, then the national master; derive all 28-state/8-UT views from that versioned graph.
- Publish model cards with source snapshot dates, licences, imputation confidence, relaxation level, missing components, validation results and prohibited interpretations.

## Final recommendation

**Proceed conditionally with Phase 0 and a single regionalised state pilot.** The research method transfers well at the level of architecture and software, and this repository already has much of the solver/scenario machinery. The main work is not “put India on a map”; it is constructing and validating the absent OSM-to-electrical-model stage, replacing US calibrations, preserving cross-state/HVDC physics, and building a lawful provenance chain.

A successful deliverable should be named along the lines of **“India OSM-derived reconstructed transmission benchmarks for GridSFM”**. It should not be called the Indian grid, a digital twin, an operational model, or a tool for real dispatch/security decisions.
