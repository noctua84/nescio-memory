---
name: glyphctl-plugin-conventions
description: Synthetic fixture. Invented standing conventions for plugins of the fictional `glyphctl` CLI — naming, exit codes, configuration precedence, logging, errors, and the testing rule. Six level-two sections, roughly three kilobytes.
type: context
---

Conventions for writing plugins for `glyphctl`, a command-line tool that exists
only in this test fixture. Nothing here describes a real project; the value of
the file is its shape — six level-two sections in the one-and-a-half to four
kilobyte band — not its content.

## Plugin naming

A plugin's distribution name is `glyphctl-<verb>` and its entry point is the
bare verb. The verb is a single lower-case word in the imperative: `render`,
`lint`, `bundle`. A two-word verb is a sign that the plugin is doing two things
and should be two plugins, each independently installable and independently
removable.

## Exit codes

Zero means the work was done. One means the user asked for something
impossible and the message explains what. Two means the plugin itself broke and
the message is a stack trace the user should report. Three is reserved for "the
inputs were not ready yet", which is a distinct outcome from failure because
callers in a loop want to retry it rather than give up.

Never reuse a code for a second meaning. A caller that branches on an exit code
has no way to tell two meanings apart, and the branch silently takes the wrong
arm the first time the second meaning occurs.

## Configuration precedence

Precedence runs, highest first: an explicit command-line flag, an environment
variable, the per-project config file, the per-user config file, the built-in
default. A plugin resolves this order once, at startup, and stores the result;
it does not consult the environment again later, because a value that can change
mid-run is a value no log line can explain.

Every resolved setting is printed when the plugin runs with `--explain`,
together with the layer it came from. A setting whose origin cannot be named is
a setting nobody can debug.

## Logging

Write progress to standard error and results to standard output, so that piping
the output into another tool never swallows the progress and never corrupts the
data. One line per unit of work, prefixed with the verb. No spinners when the
stream is not a terminal.

Log the error where it happens, with the identifier of the thing being worked
on. A message that says only "failed" forces the reader to reconstruct which of
four hundred inputs it was about.

## Error reporting

An error message names the input, the expectation and the observed value, in
that order, and nothing else. It does not apologise, does not suggest that the
user may have made a mistake, and does not wrap the underlying exception in a
new one that only repeats its text with different punctuation.

If recovery is possible, the message says what to do. If it is not, the message
says so plainly, which is more useful than an invitation to retry something that
cannot succeed.

## Tests are exempt from the duplication rule

The rule against duplicated logic applies to plugin code. Tests are the
deliberate exception: keep each test self-contained even when that means
repeating its setup. A shared helper that quietly couples a dozen otherwise
unrelated tests is a worse failure mode than a little repetition, because one
edit to the helper changes the meaning of every test that uses it without any
of them mentioning the change.
