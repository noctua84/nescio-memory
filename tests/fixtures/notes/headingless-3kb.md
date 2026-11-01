---
name: why-a-three-kilobyte-headingless-note-stays-whole
description: The control case for the split decision - same size band as a structured note, but with no structure to exploit
type: context
---
A note of this size sits squarely in the band where the split decision is
interesting. It is comfortably past the soft target of fifteen hundred
characters, so an implementation that keyed only on length would cut it into
several passages. It is comfortably under the hard cap of four thousand, so the
packing branch has no reason to touch it either. What decides its fate is the
one thing length cannot tell you: whether the author gave the document any
internal structure worth cutting at.

Here the answer is no. There is not a single level-two or level-three heading in
this file. Every paragraph runs into the next as continuous prose, the way a
long comment, a chat transcript or a page of extracted text does. Cutting such a
document at an arbitrary character offset would produce passages that begin
mid-thought and end mid-thought, and a retrieval system that returns those is
worse than one that returns the whole note, because the reader now has to go
fetch the surrounding text anyway and has lost the sentence that explained why
any of it mattered.

Contrast that with a note of exactly the same length that carries four
level-two headings. There, the author has already done the work of saying where
one topic stops and the next begins, and honouring those boundaries produces
passages that each answer a question on their own. The measurement behind this
rule is concrete: of the four hundred and fifty-eight notes in the corpus at or
under four kilobytes, sixty-four carry three or more level-two headings, and
every one of those sixty-four falls in the one-and-a-half to four kilobyte
band. A threshold set at four kilobytes would have flattened all sixty-four of
them into single multi-topic passages, which is precisely the failure this
model exists to remove.

So structure, not size, is what licenses a split, and size alone only ever
forces one when a unit would otherwise exceed the hard cap. This fixture is the
control that holds the rule honest: if someone reintroduces a length-only
threshold, the assertion that this document yields exactly one passage with no
heading path is the test that goes red first, and it says in its name exactly
what was lost.

One more paragraph of prose, so that the file lands near three kilobytes rather
than two, keeping it in the same band as its structured counterpart and making
the comparison between the two fixtures a fair one.

And a final one, for the same reason: the two fixtures are only a useful pair
while they are genuinely the same size, so that no one reading a failure can
explain it away as a size difference rather than the structural difference it
actually is. Three kilobytes of prose, no headings, one passage.
