# Changelog

## [0.2.0](https://github.com/noctua84/nescio-memory/compare/v0.1.0...v0.2.0) (2026-09-22)


### ⚠ BREAKING CHANGES

* routes moved from /search and /ingest to /api/v1/search and /api/v1/ingest. IngestResponse now serializes as {"status", "file", "ingested"} instead of {"status", "file_path", "chunks_ingested"}.

### Features

* [impl] add open api spec exporter, spec files ([a199572](https://github.com/noctua84/nescio-memory/commit/a199572ba97150c6c4762e62ea4ac72018e01702))
* [impl] add semantic memory API with search and ingest endpoints ([e2bf367](https://github.com/noctua84/nescio-memory/commit/e2bf3675ad2805e40a23d99f40484db32183c3ab))
* [impl] make in-process embeddings an optional extra ([9955d33](https://github.com/noctua84/nescio-memory/commit/9955d3376d366404c81f68d9ea1a44cc6e12f28c))
* [impl] report the active embedding backend from /health ([8f9e799](https://github.com/noctua84/nescio-memory/commit/8f9e799d1bdc63f310870295329d32540ae021b1))


### Bug Fixes

* [impl] filter search on the repo_name column ([daec173](https://github.com/noctua84/nescio-memory/commit/daec173401fa88a3db361ce855171e4b1d45e474))


### Documentation

* [chore] add agent context file ([683065c](https://github.com/noctua84/nescio-memory/commit/683065c4a64c238df8fcfc9c2d0b4ddee39c9e3b))
* [chore] add README, MIT license and project metadata ([f936f10](https://github.com/noctua84/nescio-memory/commit/f936f10fc88067750b15fbd3a93ab249a1366621))
* [chore] align README and QWEN.md with the app package refactor ([6b0fdb0](https://github.com/noctua84/nescio-memory/commit/6b0fdb064fa9bed838b801940170b482370852c1))


### Code Refactoring

* [impl] restructure into an app package behind /api/v1 ([1108d7b](https://github.com/noctua84/nescio-memory/commit/1108d7b7e536619afc4207d3bc597dd9f61b9314))
