# Changelog

All notable changes to this project will be documented in this file.

Change that doesn't affect end user should not be listed:
- CI change
- Github specific file change

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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


[Unreleased]: https://github.com/pasqal-io/qubo-solver/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.1...v0.7.0
[0.6.2]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.5.0...v0.6.0
