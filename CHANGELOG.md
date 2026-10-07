# Changelog

## [0.5.0](https://github.com/noctua84/nescio-memory/compare/v0.4.0...v0.5.0) (2026-10-07)


### Features

* [impl] expand search hits to their surrounding context ([400a531](https://github.com/noctua84/nescio-memory/commit/400a5313893afd063199ca99ee25c3b11f06eadf))
* expand search hits to their surrounding context ([b04b807](https://github.com/noctua84/nescio-memory/commit/b04b8070171baadc879e832db2e42e06318f173e))


### Bug Fixes

* [impl] attach traceback to the DataError warning log ([53aa640](https://github.com/noctua84/nescio-memory/commit/53aa64071936425fbd14867eba1a5ef2c5d6a315))
* [impl] log embedding backend failures before responding ([c675e5f](https://github.com/noctua84/nescio-memory/commit/c675e5fd1a66bd2c267d95089c48cfb3236fd8ad))
* [impl] map sqlalchemy.exc.DataError to a 400 client-error response ([449fbba](https://github.com/noctua84/nescio-memory/commit/449fbbacc6bf49efe62483af93ce051e5161a6b8))
* [impl] reject float32-overflowing and huge-int embedding components ([0a86667](https://github.com/noctua84/nescio-memory/commit/0a86667c3f44d555757b60897843c16255abceda))
* [impl] reject non-finite embedding components in _validated() ([ec002c6](https://github.com/noctua84/nescio-memory/commit/ec002c69751882efe124aff32027aaecf651891a))
* [impl] report a backend emitting unstorable embedding values ([8c94e93](https://github.com/noctua84/nescio-memory/commit/8c94e937f68725603eea7e08909938924cea9b00))
* [impl] set the embedding dimension to the model's real width (1024) ([d8335a2](https://github.com/noctua84/nescio-memory/commit/d8335a2eaec629807d6b5951a369657f9ab1a469)), closes [#36](https://github.com/noctua84/nescio-memory/issues/36)
* map sqlalchemy.exc.DataError to a 400 instead of a bare 500 ([705e50c](https://github.com/noctua84/nescio-memory/commit/705e50c047827e192bc0f78a7c566bfefbe5348f))
* reject embedding values pgvector cannot store, and make the failure visible ([8c0659b](https://github.com/noctua84/nescio-memory/commit/8c0659bef7ae0bb2cc082bdfc1289663af549ba5))
* set the embedding dimension to the model's real width (1024) ([158cd05](https://github.com/noctua84/nescio-memory/commit/158cd057ff64ba3910306f6ae19d5670b551ab43))


### Documentation

* [docs] narrow the unmapped-errors limitation for the DataError handler ([dfbb0d0](https://github.com/noctua84/nescio-memory/commit/dfbb0d02a98f4158ab11dbac346569045d582b2a))
* [docs] specify context expansion in search ([b5eaad2](https://github.com/noctua84/nescio-memory/commit/b5eaad269319303534dc6d0c83d56b9e886091ad))

## [0.4.0](https://github.com/noctua84/nescio-memory/compare/v0.3.1...v0.4.0) (2026-10-02)


### Features

* [impl] cap search with a configurable statement_timeout ([#17](https://github.com/noctua84/nescio-memory/issues/17)) ([f1c4c92](https://github.com/noctua84/nescio-memory/commit/f1c4c92df63e431a80dfe325d4438f4c036944f1))
* cap search with a configurable statement_timeout ([#17](https://github.com/noctua84/nescio-memory/issues/17)) ([2933378](https://github.com/noctua84/nescio-memory/commit/29333786fbd34cb1e58b22099e7ed1797157abfe))


### Bug Fixes

* [fix] accurate timeout warning and a search-path timeout test ([#17](https://github.com/noctua84/nescio-memory/issues/17)) ([1c0a632](https://github.com/noctua84/nescio-memory/commit/1c0a6326e794454001423246458377d3808a0d22))
* [fix] close stabilization items 2-7 ([802e385](https://github.com/noctua84/nescio-memory/commit/802e385fcf1858994d71edaab3e0ac7eecd1bca5))
* [fix] define HTTP behaviour when Ollama or PostgreSQL fail ([b827d63](https://github.com/noctua84/nescio-memory/commit/b827d6345aedf11e046ff22325528ff55cc8ed96))
* [fix] detect statement timeouts via psycopg 3 after the driver switch ([#17](https://github.com/noctua84/nescio-memory/issues/17)) ([047ec64](https://github.com/noctua84/nescio-memory/commit/047ec64b652e2493b611cf7a384ed6f509b70cc6))
* [fix] pin bare postgresql:// DATABASE_URL to psycopg2 for SQLAlchemy 2.1 ([f1cecb6](https://github.com/noctua84/nescio-memory/commit/f1cecb6aa6188765d39b371834e2127b767e7a39))
* [fix] stop HNSW post-filter starvation in /api/v1/search ([#15](https://github.com/noctua84/nescio-memory/issues/15)) ([d4eddd6](https://github.com/noctua84/nescio-memory/commit/d4eddd6bd7f93a9591fc70cb2da94fe97893648d))
* [fix] switch PostgreSQL driver to psycopg 3 for SQLAlchemy 2.1 ([9ae8709](https://github.com/noctua84/nescio-memory/commit/9ae87091f9e3fc2aabcafa83cb3157d556ce1ea3))
* [fix] switch PostgreSQL driver to psycopg 3 for SQLAlchemy 2.1 ([5d5b601](https://github.com/noctua84/nescio-memory/commit/5d5b6011739030f70b1ca3585ccb9c2fb42540e3))
* [impl] switch PostgreSQL driver to psycopg 3 ([c6843ab](https://github.com/noctua84/nescio-memory/commit/c6843abe1dc52d7dcb0097a5a0cde7429e272e15)), closes [#16](https://github.com/noctua84/nescio-memory/issues/16)
* fail loudly when search's HNSW settings do not take effect ([#22](https://github.com/noctua84/nescio-memory/issues/22)) ([20783d2](https://github.com/noctua84/nescio-memory/commit/20783d2b6251476daeab295467394021b3cff080))
* guard psycopg 3 driver resolution with a regression test ([6f0bc2c](https://github.com/noctua84/nescio-memory/commit/6f0bc2c411860e268040153050871dfb87c09875))


### Documentation

* [chore] namespace plan scratch files per plan and ignore .superpowers/ ([0cf7653](https://github.com/noctua84/nescio-memory/commit/0cf76530596334e4d6ce83c006cc763cc357c9a9)), closes [#21](https://github.com/noctua84/nescio-memory/issues/21)
* [docs] QA audit report for the statement_timeout change ([#17](https://github.com/noctua84/nescio-memory/issues/17)) ([0b0a225](https://github.com/noctua84/nescio-memory/commit/0b0a22560b21a8649af41063b8a95c85be95232d))
* namespace plan scratch files per plan and ignore .superpowers/ ([1ec000b](https://github.com/noctua84/nescio-memory/commit/1ec000beded708103b8fea286c8c38824c1c3147))

## [0.3.1](https://github.com/noctua84/nescio-memory/compare/v0.3.0...v0.3.1) (2026-09-28)


### Bug Fixes

* [fix] create the learnings.file_path index the model declares ([4c8968d](https://github.com/noctua84/nescio-memory/commit/4c8968ddd0655f9fdf6648cd25e1a0f47bab6c00))


### Documentation

* document the API key auth flow and bring the README back in line with the code ([#8](https://github.com/noctua84/nescio-memory/issues/8)) ([d36522a](https://github.com/noctua84/nescio-memory/commit/d36522a3eaabd3460c62846e3500bb4bc2a753d4))

## [0.3.0](https://github.com/noctua84/nescio-memory/compare/v0.2.1...v0.3.0) (2026-09-27)


### Features

* [impl] authenticate requests with hashed API keys ([93eb5db](https://github.com/noctua84/nescio-memory/commit/93eb5db4f3d69d7222b7979425fee1e55b718636))
* [impl] database setup with sqlalchemy ([b552fb6](https://github.com/noctua84/nescio-memory/commit/b552fb6d4ca37e15b3a1d7acbe77aee354cd1a4b))
* [impl] scope learnings to the authenticated client ([4694970](https://github.com/noctua84/nescio-memory/commit/4694970e437c501a46ebb42418183a3bf07c355a))
* [impl] switch the embedding column to 384 dimensions ([f0bcb10](https://github.com/noctua84/nescio-memory/commit/f0bcb10ca16573351335c5e379099c1d2631a37f))


### Bug Fixes

* [fix] reject absolute and traversing file_path values on ingest ([69ac42e](https://github.com/noctua84/nescio-memory/commit/69ac42efafbc9a3637dcde395023fde3d5be9e32))
* [fix] type the embedding column as list[float], not dict ([752b4d8](https://github.com/noctua84/nescio-memory/commit/752b4d8eefc0f6fa4958ba3e7584456bc7577c57))


### Documentation

* [docs] add the design spec for the test harness ([0d8703f](https://github.com/noctua84/nescio-memory/commit/0d8703fbbfdf9e208063073d152a12b711ddee40))
* [docs] add the test infrastructure implementation plan ([64c2d5d](https://github.com/noctua84/nescio-memory/commit/64c2d5db2eed08b1462c7854867938c5862051f6))

## [0.2.1](https://github.com/noctua84/nescio-memory/compare/v0.2.0...v0.2.1) (2026-09-22)


### Bug Fixes

* [impl] stop chunk_text from looping forever on a bad window config ([ba72898](https://github.com/noctua84/nescio-memory/commit/ba72898afbfd8d037b6021a476ddae5afc907796))

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
