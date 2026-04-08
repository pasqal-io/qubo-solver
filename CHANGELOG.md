# Changelog

All notable changes to this project will be documented in this file.

Change that doesn't affect end user should not be listed:
- CI change
- Github specific file change

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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


[Unreleased]: https://github.com/pasqal-io/qubo-solver/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/pasqal-io/qubo-solver/compare/v0.5.0...v0.6.0
