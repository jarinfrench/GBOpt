Phase 4 of the Zhang UO2 clean-boundary remediation is complete.

The existing geometry audit has been promoted into a reusable, strict, topology-aware feasibility validator operating directly on `BicrystalState`. Validation now produces deterministic `invalid`, `infeasible`, `warning`, or `feasible` decisions using configurable species-aware duplicate, hard-contact, and warning-contact thresholds; local per-interface void criteria; and separate slab surface, vacuum, fixed-region, and buffer-region checks.

Raw metrics and reason codes are retained independently of the effective status. Expert overrides require an explicit reason, remain visible in the serialized report, and cannot override structurally invalid states. Periodic bicrystals evaluate both coupled physical interfaces, while intentional slab vacuum is excluded from grain-boundary void measurements.

Generation now persists the complete feasibility policy and report, including deterministic policy and report hashes, and verifies those hashes before reusing cached outputs. The validator does not modify coordinates or perform translation search, termination enumeration, target-property evaluation, MC/GA integration, external-relaxation persistence, or optimizer checkpointing.

Regression coverage includes synthetic topology and threshold cases, mixed periodic/fixed boundary conditions, slab validation, deterministic serialization, override behavior, and real reduced Zhang cases. The full repository test suite passes.

The next task is Phase 5: implement a topology-aware rigid-translation primitive for `BicrystalState`. It must support explicit three-component lab-frame displacement, wrap only periodic axes using the actual lower and upper bounds, preserve identity and topology metadata, update cumulative relative-translation provenance deterministically, and return a new state without performing candidate search or optimizer integration.
