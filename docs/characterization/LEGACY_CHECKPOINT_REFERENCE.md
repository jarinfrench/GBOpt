# Legacy checkpoint behavior reference

## Reference source

The older checkpoint-enabled archive is a behavioral reference only:

```text
gbopt_logging_source.tar.gz
2ea4ba4cfa60a4ae432d6af952acd279d6282dd50457cd2828acd89f951ae2fc
```

No code from that archive is included in F0.

## Behavior to audit in CP0

The legacy implementation and tests demonstrate these capability groups:

1. Optional disabled/null checkpointing that creates no checkpoint file.
2. JSON and pickle serialization, schema envelopes, interval saves, and unconditional
   final saves.
3. MC restoration of accepted structure, RNG state, temperature, rejection count,
   objective history, accepted indices, operation history, run ID, minimum steps, and
   cooldown behavior.
4. MC continuation after a previously completed run by increasing `max_steps`.
5. GA generation-boundary snapshots and restoration of ordered population artifacts,
   lineages, objective histories, run identity, and RNG state.
6. Per-candidate intra-generation result caching that skips completed evaluations after
   interruption.
7. Batch-evaluator recovery at batch-return granularity, with an optional explicit
   finer-grained callback in the legacy implementation.
8. Publication and cleanup ordering for pending next-generation artifacts.
9. Loud failure when a checkpoint references a missing required structure artifact.
10. Continuous-versus-resumed equivalence tests for MC and GA.

## F0 representation

`tests/characterization/test_checkpoint_reference.py` contains skipped reference tests
that name these obligations. They remain skipped until checkpoint functionality is
introduced. They do not promise:

- compatibility with arbitrary pickle payloads;
- retention of the legacy dictionary schema;
- function-signature introspection as the final batch recovery API;
- reuse of legacy production modules.

CP0 must classify each behavior as preserved, externally preserved through redesign,
intentionally rejected, or optional compatibility before CP1 begins.
