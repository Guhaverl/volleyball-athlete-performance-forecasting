# Volleyball World source notes

_Last reviewed: 2026-08-11._

## Observed public routes

The project isolates these currently observable routes behind one client:

```text
https://en.volleyballworld.com/api/v1/globalschedule/competitions/{year}/{month}
https://en.volleyballworld.com/api/v1/globalschedule/{start}/{end}
https://en.volleyballworld.com/api/v1/volley-tournament/{start}/{end}/{tournament_no}
https://en.volleyballworld.com/volleyball/competitions/{slug}/{season}/players/{player_id}
```

The API routes provide competition or match context. Public player pages contain
match-by-match tables for scoring and skill categories. No stable, documented
public athlete-stat JSON contract was located during repository preparation.
Therefore:

- URLs are configurable and may change without notice.
- An authorized JSON adapter is supported through an environment URL template
  and YAML field mapping.
- The HTML parser is a fallback, covered by offline fixtures.
- Requests are cached, rate-limited, and use an identifiable user agent.
- Live-network tests should be opt-in and never run on every pull request.

## Responsible use checklist

1. Review the current site terms, robots directives, and applicable licenses.
2. Obtain permission for commercial use, bulk collection, or redistribution.
3. Store only the minimum data required for the research purpose.
4. Respect deletion, attribution, and access-control requirements.
5. Stop ingestion when route behavior or authorization changes.
