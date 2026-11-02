"""Hand-run retrieval evaluation harness. NOT a test suite -- keep it that way.

This package exists to answer one question with numbers: for a corpus of
human-curated markdown notes, does indexing each note's `name + description`
as a single retrieval key beat the service's current behaviour of chunking the
note body at CHUNK_SIZE/CHUNK_OVERLAP? That question is currently argued from
the shape of the corpus rather than measured against queries.

WHY THIS IS NOT IN tests/
-------------------------
`tests/` is hermetic and deterministic. It fakes the embedder on purpose (see
tests/fakes.py: a SHA-256-seeded unit vector) because its job is to pin the
service's *mechanics* -- the tenant WHERE clause, the vector(384) column, the
<=> operator, the chunk window -- against arithmetic the suite controls.

A deterministic fake embedder cannot answer this package's question at all. The
question is which indexing strategy retrieves better *semantically*, and a
hash-derived vector has no semantics: paraphrases land nowhere near each other,
so every strategy scores the same noise. Measuring the strategies therefore
requires the real embedding model, which means a running Ollama server, a model
download, non-determinism between model versions, and tens of seconds to
minutes per run.

Those are exactly the properties a test suite must not have. So:

  * This harness is run BY HAND, by a person, when they want a measurement.
  * It is NOT wired into CI. CI has no model.
  * It imports nothing from `tests/`, and `tests/` imports nothing from here.
    The duplication that follows from that (this package has its own stub
    embedder) is deliberate: the two directories must be able to drift apart,
    because they are answering different questions.

If you blur the two -- fake the embedder here to make it fast and
deterministic, or import this package from a test -- the harness stops
measuring anything and becomes an expensive way to assert that arithmetic
works. It is worthless at that point. Do not do it.

WHAT THIS HARNESS DOES NOT MEASURE
----------------------------------
It holds every vector in memory and computes cosine similarity directly, with
no database. That is deliberate -- it isolates the result to the indexing
strategy -- but it means the numbers are an upper bound: production retrieves
through a pgvector HNSW index, which is approximate and post-filtered by
client_name, and loses some recall of its own. This harness says nothing about
that loss. See LearningRepository.search for the production index's behaviour.
"""
