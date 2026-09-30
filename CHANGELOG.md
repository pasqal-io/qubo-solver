# Changelog

All notable changes to this project will be documented in this file.

Change that doesn't affect end user should not be listed:
- CI change
- Github specific file change

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.9.0] - 2026-10-05

This release is a major refactoring towards v1.0 and contains many breaking API changes.

### Added
- Add `extract_qubo` to reconstruct a QUBO from a register and a drive. ([#225](https://github.com/pasqal-io/qubo-solver/pull/225))
- Handle negative off-diagonal QUBO coefficients through GLPK-based bit-flip preprocessing, with a zeroing fallback. ([#229](https://github.com/pasqal-io/qubo-solver/pull/229), [#238](https://github.com/pasqal-io/qubo-solver/pull/238))
- Add the Local-Energy-Scale drive shaper (`DriveType.LOCAL_ENERGY_SCALE`). ([#258](https://github.com/pasqal-io/qubo-solver/pull/258))
- Add first-improvement and greedy-sweep strategies to the bit-flip post-processing local search, and a time budget shared across the batch. ([#259](https://github.com/pasqal-io/qubo-solver/pull/259), [#277](https://github.com/pasqal-io/qubo-solver/pull/277))
- Report per-bitstring visit counts in simulated annealing solutions. ([#253](https://github.com/pasqal-io/qubo-solver/pull/253))
- Add `Solution.check_consistency()` to validate solutions. ([#240](https://github.com/pasqal-io/qubo-solver/pull/240))
- Add a `("quantile", q)` option for the greedy layout `max_possible_term`, and start BLaDE from a multi-dimensional scaling (MDS) of the QUBO. ([#315](https://github.com/pasqal-io/qubo-solver/pull/315))
- Add an L2 norm option (`p=2`) to the greedy layout embedding cost function. ([#317](https://github.com/pasqal-io/qubo-solver/pull/317))
- Support Python 3.13 and 3.14. ([#279](https://github.com/pasqal-io/qubo-solver/pull/279))

### Changed
- Restructure the `qubosolver` package and overhaul the documentation. ([#220](https://github.com/pasqal-io/qubo-solver/pull/220), [#279](https://github.com/pasqal-io/qubo-solver/pull/279))
- Make embedding and drive shaping adimensional, so the register and the drive share a single unit system. ([#223](https://github.com/pasqal-io/qubo-solver/pull/223))
- Rename algorithms, config fields and types to more meaningful names (e.g. `HeuristicDriveShaper` → `ProportionalDiagonalDriveShaper`, `OptimizedDriveShaper` → `BayesianSearchDriveShaper`, `SingleSolution` → `Candidate`). ([#249](https://github.com/pasqal-io/qubo-solver/pull/249), [#279](https://github.com/pasqal-io/qubo-solver/pull/279))
- Simplify `SolverConfig`: drop rarely-used tunables, add a single `ClassicalConfig.time_limit`, enable preprocessing and post-processing by default, and use the local-energy-scale drive shaper by default. ([#310](https://github.com/pasqal-io/qubo-solver/pull/310))
- Make BLaDE the default embedding algorithm, and tune the BLaDE and greedy layout defaults. ([#293](https://github.com/pasqal-io/qubo-solver/pull/293), [#315](https://github.com/pasqal-io/qubo-solver/pull/315))
- Speed up the greedy layout embedding and drastically reduce its memory usage. ([#317](https://github.com/pasqal-io/qubo-solver/pull/317))
- Speed up simulated annealing and tabu search with incremental cost tracking. ([#275](https://github.com/pasqal-io/qubo-solver/pull/275), [#276](https://github.com/pasqal-io/qubo-solver/pull/276))
- Make `cplex` an optional dependency, installable with `qubo-solver[extras]`. ([#286](https://github.com/pasqal-io/qubo-solver/pull/286))
- Declare missing runtime dependencies (`scipy`, `pasqal-cloud`, `pulser`, `emu-mps`, `emu-sv`) and move dev/doc dependencies to dependency groups. ([#280](https://github.com/pasqal-io/qubo-solver/pull/280))
- Require `qoolqit>=1.4.0`. ([#245](https://github.com/pasqal-io/qubo-solver/pull/245), [#279](https://github.com/pasqal-io/qubo-solver/pull/279))

### Removed
- Remove the ability to select the decomposition solver directly from `Solver`/`SolverConfig`. ([#301](https://github.com/pasqal-io/qubo-solver/pull/301))
- Remove the `energy_tol` parameter of simulated annealing. ([#253](https://github.com/pasqal-io/qubo-solver/pull/253))

### Fixed
- Fix bitstring/cost mispairing in simulated annealing. ([#231](https://github.com/pasqal-io/qubo-solver/pull/231))
- Use a different start for each independent tabu search run. ([#233](https://github.com/pasqal-io/qubo-solver/pull/233))
- Round CPLEX bitstrings instead of truncating them. ([#240](https://github.com/pasqal-io/qubo-solver/pull/240))
- Handle empty, single-variable and zero-variable (e.g. after preprocessing) instances across the solvers. ([#247](https://github.com/pasqal-io/qubo-solver/pull/247), [#303](https://github.com/pasqal-io/qubo-solver/pull/303), [#314](https://github.com/pasqal-io/qubo-solver/pull/314))
- Fix runtime type checks (`QUBO_SOLVER_RUNTIME_CHECKS=1`), which were not applied to the whole package. ([#318](https://github.com/pasqal-io/qubo-solver/pull/318))

## [0.8.2] - 2026-08-20

### Fixed
- Republish of 0.8.1: the previous release failed to publish to PyPI due to a metadata-version validation error in the publish workflow. No package changes since 0.8.1. ([#267](https://github.com/pasqal-io/qubo-solver/pull/267))

## [0.8.1] - 2026-08-19

### Fixed
- Clamp `qoolqit`, `pulser`, `pulser-pasqal`, `emu-base`, `emu-sv`, and `emu-mps` upper bounds: newer releases of these packages break `qubo-solver` 0.8.0, and the incompatibility will only be fixed in v1. ([#263](https://github.com/pasqal-io/qubo-solver/pull/263))

## [0.8.0] - 2026-06-29

### Added
- Automatic local emulator selection based on qubit count (QutipBackendV2 / SVBackend / MPSBackend). ([#155](https://github.com/pasqal-io/qubo-solver/pull/155), [#173](https://github.com/pasqal-io/qubo-solver/pull/173))
- Automatic remote emulator selection (keeps `EMU_FREE` as default due to pricing considerations). ([#180](https://github.com/pasqal-io/qubo-solver/pull/180))
- Add metadata for cloud analytics. ([#171](https://github.com/pasqal-io/qubo-solver/pull/171))
- Add time limit support for classical Simulated Annealing (SA) (runtime budget-based stopping). ([#186](https://github.com/pasqal-io/qubo-solver/pull/186))
- Add time limit support for Tabu Search (runtime budget-based stopping). ([#190](https://github.com/pasqal-io/qubo-solver/pull/190))

### Changed
- Change defaults: `min_distance=1.001`, `device=AnalogDeviceWithDMM` (requires `qoolqit 1.1.1`). ([#196](https://github.com/pasqal-io/qubo-solver/pull/196))
- Update waveform imports in anticipation of upcoming `qoolqit` changes. ([#204](https://github.com/pasqal-io/qubo-solver/pull/204))
- Bump `numpy` version requirement to `>=2`. ([#205](https://github.com/pasqal-io/qubo-solver/pull/205))

### Docs
- Fix QPU tutorial number 02. ([#209](https://github.com/pasqal-io/qubo-solver/pull/209))

## [0.7.2] - 2026-06-05

### Docs
- Remove `polyfill.io` from GitHub Pages documentation to prevent intrusive pop-up/redirect behavior for some visitors ([#175](https://github.com/pasqal-io/qubo-solver/pull/175))

## [0.7.1] - 2026-05-22

### Changed
- Update v0.7 documentation, and improve docstrings across the library ([#159](https://github.com/pasqal-io/qubo-solver/pull/159))

## [0.7.0] - 2026-05-07

### Added
- Add remote job support using the new Qoolqit Job API: send jobs to Pasqal Cloud, retrieve job IDs, and fetch results asynchronously ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))
- Add partial serialization of the solver, allowing it to be restored and continue post-processing once results are fetched from the cloud ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))

### Changed
- Bump `qoolqit` dependency to `>=1.1` (from `==0.3.1`); use `qoolqit[extras]` instead of `qoolqit[solvers]` ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))
- Use Qoolqit's adimensionalization ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))
- Default solver mode is now quantum (`use_quantum=True` by default in `SolverConfig`) ([#154](https://github.com/pasqal-io/qubo-solver/pull/154))
- Extend supported Python versions to `<=3.14` (was `<3.13`) ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))

### Removed
- Remove BLaDE embedding algorithm: it has been moved to Qoolqit ([#156](https://github.com/pasqal-io/qubo-solver/pull/156))

## [0.6.2] - 2026-06-05

### Docs
- Remove `polyfill.io` from GitHub Pages documentation to prevent intrusive pop-up/redirect behavior for some visitors ([#175](https://github.com/pasqal-io/qubo-solver/pull/175))

## [0.6.1] - 2026-04-14

### Fixed
- Fix documentation rendering issues (wrong backslashes and math formulas not displaying correctly) ([#146](https://github.com/pasqal-io/qubo-solver/pull/146), [#149](https://github.com/pasqal-io/qubo-solver/pull/149))

## [0.6.0] - 2026-04-07

### Added
- Add new pulse shaping heuristic ([#107](https://github.com/pasqal-io/qubo-solver/pull/107))
- Add citation file ([#112](https://github.com/pasqal-io/qubo-solver/pull/112))
- Better defaults for the greedy embedder and `SolverConfig` default fields ([#97](https://github.com/pasqal-io/qubo-solver/issues/97), [#137](https://github.com/pasqal-io/qubo-solver/pull/137))

### Fixed
- Fix SA classical solver multiplying by 2 the cost of each solution by reverting [#89](https://github.com/pasqal-io/qubo-solver/pull/89) ([#113](https://github.com/pasqal-io/qubo-solver/pull/113))
- Fix wrong preprocessing implementation behavior ([#96](https://github.com/pasqal-io/qubo-solver/issues/96), [#123](https://github.com/pasqal-io/qubo-solver/pull/123))
- Fix drive-shaping not using the reduced QUBO in pre-processing ([#123](https://github.com/pasqal-io/qubo-solver/pull/123))
- Fix heuristic drive shaper scaling and hardware constraint handling ([#139](https://github.com/pasqal-io/qubo-solver/pull/139))

### Removed
- Remove `roof_duality_fixing` pre-processing step and its `maxflow` dependency ([#120](https://github.com/pasqal-io/qubo-solver/pull/120))
- Remove adiabatic drive shaper ([#138](https://github.com/pasqal-io/qubo-solver/pull/138))

### Changed
- Fix all mypy type errors ([#117](https://github.com/pasqal-io/qubo-solver/pull/117))

### Docs
- Remove references to D-Wave ([#118](https://github.com/pasqal-io/qubo-solver/pull/118))


[Unreleased]: https://github.com/pasqal-io/qubo-solver/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/pasqal-io/qubo-solver/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.1...v0.7.0
[0.6.2]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.5.0...v0.6.0
