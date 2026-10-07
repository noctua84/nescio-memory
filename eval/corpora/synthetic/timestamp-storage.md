---
name: Timestamp storage, offsets versus zones, and the recurring event problem
type: reference
---
<!--
This note deliberately has NO `description` in its frontmatter. The real corpus
has a handful of notes in the same state, and the `summary` strategy's name-only
fallback has to be exercised by the committed fixture corpus rather than only
discovered against a private brain. Do not "fix" it by adding one.
-->
Instants are stored as an absolute point in time with an offset, in a column type that preserves that, and never as a naive local time with the zone left implicit elsewhere. An instant with no zone information is ambiguous for one hour every year and non-existent for another, and the code that eventually has to interpret it will be far away from the code that wrote it. This part is uncontroversial and mostly a matter of discipline.

The part that is genuinely hard is that an offset is not a zone, and storing an offset is only sufficient for things that already happened. An offset tells you the difference from universal time at one moment; a zone tells you the rule that produces offsets, including when that rule changes. For a past event these are equivalent, because the offset that applied is settled and will not be revised. For a future event they are not, because the rule can change between now and then, and governments do change these rules with a few months of notice and occasionally less.

This is why a recurring event cannot be stored as a series of instants. If someone schedules a weekly meeting for nine in the morning in a particular zone, what they mean is nine in the morning local time, every week, whatever the offset turns out to be. Storing the next fifty occurrences as absolute instants bakes in today's offset rules, and when the zone's rules change, every occurrence after the change is off by an hour. The correct storage is the local wall-clock time, the zone identifier, and the recurrence rule, with instants computed at read time from the current rule set. The identifier must be the full region name rather than an abbreviation, because abbreviations are ambiguous across regions and carry no rule.

The consequence most people find surprising is that a stored future event's absolute time is not stable, and anything that cached or published it has to be able to be corrected. We refresh derived instants whenever the time zone database is updated, which happens several times a year, and that refresh is a scheduled job rather than a manual response to a news item.

Two smaller rules. A date with no time, such as a birthday or a contract date, is a plain date and must not be promoted to an instant at midnight in any zone, because midnight in one zone is the previous day in another and the date then shifts depending on who is looking. And a duration between two instants is always computed on the absolute values, never by subtracting wall-clock times, which differ by an hour across a transition and have produced both negative durations and a shift that silently double-counted an hour of billable time.
