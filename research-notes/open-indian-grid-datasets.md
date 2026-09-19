# Open datasets for modelling the Indian electricity grid

Checked 9 September 2026. Sources below are primary project, repository, or government pages.

## Short answer

There is **no verified open national or state-wise Indian dataset equivalent to PGLib-OPF or the GridSFM dataset**: i.e. a detailed transmission bus/branch model with AC parameters, generator limits/costs, loads, and many solved AC-OPF operating points. The useful open material is complementary rather than complete.

The best practical stack is:

1. **GridPath-India** for state-zone demand, projects, costs, renewable profiles, and inter-state transfer constraints.
2. **PyPSA-Earth + current OpenStreetMap data** to reconstruct the physical high-voltage graph.
3. **CEA/NPP/data.gov.in and the Global Transmission Database** to calibrate state totals and inter-state corridors.
4. Generate and validate AC-OPF cases and GridSFM labels locally.

## Ranked candidates

| Rank | Dataset/model | India coverage and format | What it contains | License/access/freshness | Fit for GridSFM |
|---|---|---|---|---|---|
| 1 | [GridPath-India](https://datadryad.org/dataset/doi%3A10.5061/dryad.dz08kpsbm) | 34 load zones; native GridPath CSV inputs; 252.52 MB | Interstate transmission constraints, hourly demand, existing/planned/candidate generation and storage, more than 1,300 candidate wind/solar sites, capacity factors, costs, reliability and policy scenarios; 2020–2050 with two representative days per month | Published 8 Jan and updated 9 Jan 2026; direct ZIP downloads. Dryad publishes accepted datasets under [CC0](https://datadryad.org/publication_policy). | **Best Indian calibration/planning source, but not an AC network.** Transmission is zonal transfer capacity, not physical lines with R/X/B, taps, voltage limits, or nodal OPF labels. |
| 2 | [PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth) | Configurable for India; outputs a PyPSA network, normally CSV/intermediate GIS and NetCDF | Workflow downloads OSM power assets, cleans lines/cables/substations, builds topology, assigns standard line types, models transformers and converters, and can add demand/generation/time series. The current [default configuration](https://github.com/pypsa-meets-earth/pypsa-earth/blob/main/config.default.yaml) uses a 51 kV threshold and explicitly exposes topology-cleaning and electrical-parameter assumptions. | Active public repository. Code/config carry open SPDX licensing, while OSM-derived database outputs remain subject to ODbL. Maintainers previously said India could run but status was uncertain and preprocessing could take 1–2 days ([project discussion](https://github.com/pypsa-meets-earth/pypsa-earth/discussions/505)). | **Best reproducible route to a physical India model.** Not a prevalidated packaged Indian case; topology completeness and electrical parameters are inferred, and no measured AC-OPF labels are supplied. |
| 3 | [Global Transmission Database v1.1](https://zenodo.org/records/10870602) | India included at subnational/regional level; four downloadable CSVs plus mapping CSV | Existing and planned transfer capacities in MW, including land and subsea pathways; India is one of the large countries with regional-level data. | Open Zenodo release, version 1.1, 25 Mar 2024; files total 418.6 kB. Processing code is [MIT-licensed](https://github.com/Electricity-Transmission-Database/electricity-transmission-database). | **Good for interstate/interface calibration.** Corridors connect regions, not physical buses; no R/X/B, transformer data, dispatch labels, or actual route geometry. |
| 4 | [TransitionZero 24/7 CFE India stock model](https://github.com/transition-zero/tza-google-cfe) | Five India regional nodes in PyPSA; reproducible source and stock model | Hourly generation, storage, demand and regional interconnectors for system-level CFE scenarios. The repository describes its India model as openly available and provides build/solve commands. | Public, MIT-licensed repository; actively developed in 2026. | Useful for aggregate dispatch and scenario comparison, but five buses are far too coarse for state-wise topology or GridSFM. It is linear optimisation, not a detailed AC case. |
| 5 | [India SCED 270 GW](https://github.com/SANKETIK-SANKHYAKI/india-sced-270GW) | Five ISTS regions; GAMSpy model and CSVs; one 24-hour day (21 May 2026) | Regional demand, generator stack, renewable profiles, TTC limits, dispatch, flows, congestion and regional LMP results. | Public MIT repository, dated May 2026; inputs and outputs are directly downloadable. | Useful as a recent regional dispatch/flow validation scenario. It is a five-node LP transport model, not security-constrained in the branch-contingency sense and not AC bus/branch data. |
| 6 | [MATPOWER `case22`](https://github.com/MATPOWER/matpower/blob/master/data/case22.m) | One real 22-bus, 11 kV radial feeder described as a small agricultural distribution area in eastern India; MATPOWER v2 `.m` | Bus P/Q loads, branch R/X, voltage bounds, one slack generator and an added generator cost. | Bundled in current MATPOWER and directly downloadable. Important caveat: MATPOWER's software license does not automatically cover all included cases; follow the case citation and verify reuse terms. | Valid for parser/power-flow smoke tests only. Branch thermal limits are disabled, it has no geography or time series, and it is not a transmission/state model. |
| 7 | [IIT Kanpur reduced NRPG 246-bus case](https://iitk.ac.in/pslab/downloads.php) | Reduced Northern Regional Power Grid, described by papers as 246 buses, 376 branches and 42 generating units | Potentially the closest historical Indian transmission power-flow case. | The IITK page lists a download, but on 9 Sep 2026 its linked `nrpg_data.pdf` returned HTTP 404. The page states “All Rights Reserved” and declares no dataset license. | **Do not plan around it unless IITK restores and licenses the data.** It is not presently a reliable open, machine-readable download. |

## Topology-only GIS sources

### OpenStreetMap

[Geofabrik's India extract](https://download.geofabrik.de/asia/india.html) provides the complete, frequently refreshed India `.osm.pbf`; power lines, cables, substations, transformers and plants can be selected by tags. OSM data is [ODbL-licensed](https://www.openstreetmap.org/copyright), requiring attribution and share-alike treatment of derivative databases. [Open Infrastructure Map's India statistics](https://openinframap.org/stats/country/IN) are a useful live completeness check, but the underlying records remain volunteered map data.

OSM has voltage, circuits, cables and operator/name tags where contributors supplied them. It does **not** guarantee electrical connectivity, busbar interpretation, line R/X/B, transformer taps, thermal ratings, generation limits, loads, costs, or operating points. It is therefore raw material for reconstruction, not an OPF case.

### Indian government-layer mirrors

The [`indian_power_infra` releases](https://github.com/ramSeraph/indian_power_infra/releases) provide compressed GeoJSONL snapshots of Indian lines, substations, transformers and power sources collected from REC/BharatMaps, Survey of India/NCOG, VEDAS and Ministry of Power GatiShakti layers. Release notes name the source and state a license for each asset.

These are valuable topology cross-checks, but the repository has no root license, some services are labelled “not-so-open,” and the uploader's per-asset license statement is not a substitute for authoritative terms from each government source. Treat this as a downloadable geospatial mirror requiring provenance review, not a safely redistributable PGLib replacement. The data also lacks a solved electrical model.

### World Bank/ESMAP Gridfinder

The [World Bank catalogue](https://datacatalog.worldbank.org/search/dataset/0038055/derived-map-of-global-electricity-transmission-and-distribution-lines) provides a global GeoPackage plus rasters under CC BY 4.0. It predicts transmission/distribution locations from night-time lights, roads and known OSM lines and reports about 70% validation accuracy at 1 km in tested countries. India is within the global coverage.

This is useful for rural electrification or identifying unmapped corridors, not for AC analysis: predicted paths are not authoritative assets and have no voltage-specific electrical parameters or bus connectivity.

## Statistical and operational complements

- The [CEA data API catalogue](https://cea.nic.in/api-for-central-electricity-authority-data/?lang=en) exposes installed capacity by state/region, supply position, transformation capacity, transmission line totals, generation, renewables and consumption. These are strong reconciliation targets, not nodal topology.
- The [CEA installed-capacity reports](https://cea.nic.in/installed-capacity-report/?lang=en) provide recurring PDF/Excel plant and state totals.
- The [National Power Portal](https://npp.gov.in/aboutus) disseminates generation, transmission, demand, supply and consumption information at India/region/state level. Access is dashboard/report oriented rather than a versioned OPF dataset.
- [data.gov.in](https://www.data.gov.in/) provides CEA and state resources in CSV/XLS/JSON/API form under the Government Open Data License–India; examples include [station-wise monthly generation](https://www.data.gov.in/catalog/power-generation) and state/UT generation. Coverage and update cadence vary by resource.

None of these government sources exposes the sensitive complete network model normally required for nodal AC power flow: bus identities, connected branches, impedances, taps, ratings, nodal load/reactive demand, generator capability curves and solved states.

## Older or currently unusable candidates

- The public [PyPSA-India 2037 study repository](https://github.com/ToeKneeShoe/PyPSA-India-2037-Study) contains GPL-2.0 study inputs, but models states as balancing-area nodes with inter-state transfer links and omits the intra-state AC grid. It is useful historical scenario material, not a detailed bus/branch benchmark, and has little maintenance or packaging.
- CEEW's [India Power System Model Inputs](https://www.ceew.in/india-power-system-model-inputs) describes downloadable inputs for a production-cost/reliability study, but the linked `India power system database - MAPS model V2.0.zip` returned 404 when checked. It also targets MAPS rather than an open MATPOWER/PowerModels case. Confirm availability and reuse rights directly with CEEW before depending on it.

## What is *not* an India PGLib substitute

- [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) itself contains IEEE, PEGASE/RTE, ACTIVSg, GOC and other cases in MATPOWER format, but no identified national Indian case.
- Other MATPOWER distribution cases bearing surnames of Indian authors are generic research feeders unless the case header explicitly establishes Indian physical provenance. `case22` is the clearest verified real-India example, and it is tiny.
- A public PyPSA workflow that *can be configured* for India is not equivalent to a validated, frozen India dataset.
- A state/region transport model with transfer limits is valuable for capacity planning but cannot supply GridSFM's node/edge features or AC solution labels without reconstruction.

## Recommendation for this project

Build one versioned national reconstruction and derive state views from it:

1. Use PyPSA-Earth/grid-builder on a dated OSM India extract; preserve OSM IDs and provenance.
2. Cross-check lines/substations against the Indian GIS mirrors and CEA transmission maps/totals.
3. Calibrate state loads, plants, costs and interface capacities using GridPath-India, CEA/NPP and GTD.
4. Replace global default electrical assumptions with documented Indian 50 Hz, 66/110/132/220/230/400/765 kV conductor and transformer assumptions.
5. Convert the validated model to PowerModels/MATPOWER, solve many AC-OPF scenarios, and export GridSFM `.pyg.json` labels.
6. Publish a data manifest separating observed, inferred and synthetic fields. Respect ODbL for the OSM-derived database and keep non-OSM validation layers separate when licenses are incompatible or unclear.

This produces a credible **synthetic/open-data-derived Indian benchmark**, but it should not be presented as the operational Indian grid.
