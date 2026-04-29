# Changelog

## 0.2.0

- Start speculation earlier at `tool_call_start` when recent args are available.
- Cancel in-flight tool-start guesses when streamed args prove them wrong.
- Add adaptive speculation pause for low observed hit-rate.
- Expand stats with tool-start launches, cancellations, wasted work, and pauses.
- Add tests for early speculation, cancellation, and adaptive pause behavior.

## 0.1.0

- Initial Python package rewrite.
- Add idempotency registry, speculation cache, predictor, runtime hooks, demos,
  and unit tests.
