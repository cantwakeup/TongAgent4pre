# Failure Taxonomy

## Formal benchmark failures

| System | Token exhaustion | Runner error | Timeout | Other |
| --- | ---: | ---: | ---: | ---: |
| Bare Simple ReAct | 8 | 1 | 0 | 0 |
| Vanilla DeepAgents | 10 | 0 | 0 | 0 |
| TongAgent Standard | 9 | 0 | 0 | 0 |

Token exhaustion is the dominant terminal failure. BrowseComp accounts for all
eight TongAgent Standard BrowseComp failures, seven Bare token failures plus
one Bare runner error, and seven Vanilla BrowseComp token failures. The
remaining token failures occur on GAIA research tasks.

The only runner error is Bare Simple ReAct on
`browsecomp-00e082ffa8ac7bd9`. Result construction raised a non-retryable
`ValidationError` because top-level tool counters did not match the
denial-time budget snapshot. The terminal failure was preserved as
`attempt-0001`; it was not rerun. This artifact also lacks token telemetry,
which prevents a complete 16-run Bare token mean.

There were no timeouts, authentication failures, global network outages, or
duplicate formal attempts. Incorrect answers and formatting mismatches remain
ordinary benchmark outcomes, not infrastructure failures.

The largest capability gap is BrowseComp completion under the 60k shared token
ceiling, not evidence gating: all three systems scored 0/8 exact matches on the
BrowseComp subset.
