# A note that carries no metadata block

This file deliberately opens with an H1 rather than a `---` fence. There is no
frontmatter to lift off, so `name`, `description` and `type` are all `None`,
and the body is the whole document.

There is no provenance marker either, so `provenance_kind` and
`provenance_date` are both `None` as well. What is left is plain prose of
roughly one kilobyte: well under the target, with no `##` or `###` heading
anywhere, so the splitter has neither a reason nor a means to cut it apart.

H1 is not a split point. That rule exists because a document's title is not a
section boundary — the text that follows a title belongs with the preamble, not
in a passage of its own labelled with the whole document's name. Were H1 a
split point, every note would begin with a one-line passage containing nothing
but its own heading, which answers no question anyone would ask.

So the expected result for this fixture is a single passage, ordinal zero,
`heading_path` of `None`, and content equal to the body.
