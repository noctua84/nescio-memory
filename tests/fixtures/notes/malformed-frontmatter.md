---
name: "a quoted scalar that is never closed
description: [unterminated, flow, sequence
type: feedback
---
The frontmatter block above is not valid YAML, so this document is treated as
having no frontmatter at all. The three delimiter lines stay in the body: the
splitter must never silently drop text it failed to parse, and it must never
raise.

A reader opening this passage sees exactly what the author wrote, broken
frontmatter included, which is the only honest thing to show when the metadata
could not be understood.
