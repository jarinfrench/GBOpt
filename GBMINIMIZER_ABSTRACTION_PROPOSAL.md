# GBMinimizer Abstraction Layer Proposal

## Executive recommendation

Introduce an abstract `GBMinimizer` base class using a **template-method plus ask/tell design**:

- The base class owns the optimization lifecycle, evaluation dispatch, run identity, result tracking, failure normalization, and common structure/manipulator construction.
- Concrete minimizers implement only algorithm policy: how candidates are proposed, how evaluated candidates update algorithm state, and when the run terminates.
- Objective evaluators, structure reconstruction, and manipulation operations remain composed collaborators rather than subclass responsibilities.
- The existing `MonteCarloMinimizer` and `GeneticAlgorithmMinimizer` become subclasses without materially changing their algorithms.
- Compatibility wrappers preserve `run_MC()`, `run_GA()`, and the current callback signatures during migration.

This implements the extension point described in the paper without prematurely depending on the interface corrections still underway.

---

## 1. Architectural intent established by the paper

The paper describes `GBMinimizer` as the component that:

1. Depends on a `GBMaker` structure.
2. Composes a `GBManipulator`.
3. Orchestrates the optimization loop.
4. Delegates structural modification to the manipulator.
5. Delegates objective evaluation to an external calculator or user callback.
6. Supports both serial and user-provided batch evaluation.
7. Maintains optimization state and aggregates results.
8. Has multiple algorithm implementations, currently Monte Carlo and genetic algorithms.

Figure 1 on page 3 reinforces this separation: candidate manipulation and property evaluation feed into algorithm-specific selection or acceptance, followed by state updates and convergence checking. The paper also explicitly identifies abstract base classes as the planned mechanism for formalizing these extension points.

The paper does **not** specify a concrete Python method contract for the future base class. The method names and data models proposed below are therefore engineering decisions inferred from the described architecture and the current implementation, rather than an API reproduced from the paper.

---

## 2. Findings from the current source

### 2.1 There is no common `GBMinimizer` class

`GBOpt/GBMinimizer.py` currently defines:

- `Mutator`
- `MonteCarloMinimizer`
- `GeneticAlgorithmMinimizer`

The two minimizers are independent concrete classes. Their common conceptual responsibilities are duplicated rather than inherited.

### 2.2 Initialization and structure reconstruction are duplicated

Both minimizers implement nearly identical `_make_initial_manipulator()` methods:

- Monte Carlo: lines 83-103
- Genetic algorithm: lines 243-255

Reloading an evaluated structure is also duplicated:

- Monte Carlo reconstructs a `GBManipulator` inline at lines 165-171.
- The GA uses `_make_manipulator_from_file()` at lines 257-264 and also constructs two-parent manipulators directly at lines 369-374.

This is exactly the type of invariant that belongs in the abstraction layer. Every optimizer needs a consistent way to move between:

- an initial structure specification;
- a mutable candidate;
- an evaluated or relaxed structure;
- the next optimizer state.

### 2.3 Evaluation is only nominally backend-agnostic

The paper describes calculator-independent scalar objective evaluation, but the concrete optimizer code currently assumes that evaluation produces a reloadable file.

The serial callback returns:

```python
objective, dump_filename
```

The batch callback returns dictionaries containing at least:

```python
{
    "energy": ...,
    "final_dump": ...,
}
```

The algorithms then create the next `GBManipulator` from `final_dump`.

Consequently, the optimizer is presently coupled to:

- filesystem persistence;
- a filename-based structure handoff;
- a result format named specifically around energy and LAMMPS dumps.

The density-optimization example demonstrates that the objective can already be something other than energy, but it must overwrite the `"energy"` field to do so. That is a sign that the evaluator contract needs neutral `objective` vocabulary.

### 2.4 The two algorithms have inconsistent public and internal behavior

| Concern | Monte Carlo | Genetic algorithm |
|---|---|---|
| Public run method | `run_MC()` | `run_GA()` |
| Return value | Minimum scalar | `(minimum scalar, dump file)` |
| Batch evaluation | No | Yes |
| Callback exceptions | Propagate | Converted to penalty |
| Invalid output file | Not normalized | Converted to penalty |
| Per-run history | `GBE_vals`, operations, accepted indices | `GBE_vals`, lineage history |
| Run identifier | Default UUID created at function-definition time | UUID created per call |
| Runtime callback kwargs | Accepted | Not exposed by `run_GA()` |

There is therefore no common contract for callers or future subclasses.

The Monte Carlo declaration:

```python
unique_id: int = uuid.uuid4()
```

also creates the UUID once when the function is defined, rather than once per run.

### 2.5 Run state and object configuration are intermingled

Fields such as `GBE_vals`, `history`, `accepted_idx`, and `operation_list` live on the configured minimizer object. Some are reset on repeated runs and others are not.

For example, `run_GA()` resets `history` but continues appending to `GBE_vals`. A reusable minimizer object should either:

- create isolated run state for each call; or
- explicitly be single-use.

A returned run-result object is preferable.

### 2.6 The current `Mutator` is not a reusable optimizer-facing abstraction

`Mutator` accepts a list of method names, silently discards names not found on the initial manipulator, and then uses a hard-coded `match` statement rather than the bound methods it stored.

It also knows operation-specific parameter generation. For example, it chooses translation magnitudes from `GBMaker` dimensions.

That may remain a useful **legacy operation adapter**, but the base minimizer should not require every future optimizer to use this exact mutation-selection mechanism.

There is also an apparent adjacent defect at lines 49-53: both `dy` and `dz` are derived from `GB.z_dim`; `dy` would normally be expected to use the corresponding y dimension. That should be addressed through focused testing, but it is not the central abstraction decision.

---

## 3. Selected design

### 3.1 Use an abstract optimizer with a concrete lifecycle

The public base class should be:

```python
class GBMinimizer(ABC):
    def run(...) -> MinimizationResult:
        ...
```

`run()` should be concrete and should enforce the shared lifecycle:

1. Create a fresh run context and run identifier.
2. Construct the initial manipulator.
3. Evaluate the initial candidate.
4. Initialize algorithm-specific state from that evaluated candidate.
5. Ask the algorithm for one or more new candidates.
6. Evaluate those candidates serially or in a batch.
7. Record normalized results and update the global best candidate.
8. Tell the algorithm about the evaluations.
9. Check algorithm-specific and global termination criteria.
10. Return a structured result.

The algorithm hooks would be approximately:

```python
@abstractmethod
def _initialize_algorithm(
    self,
    initial_result: EvaluationResult,
    run: RunContext,
) -> AlgorithmState:
    ...

@abstractmethod
def _ask(
    self,
    state: AlgorithmState,
    run: RunContext,
) -> Sequence[Candidate]:
    ...

@abstractmethod
def _tell(
    self,
    state: AlgorithmState,
    evaluations: Sequence[EvaluationResult],
    run: RunContext,
) -> None:
    ...

@abstractmethod
def _termination_reason(
    self,
    state: AlgorithmState,
    run: RunContext,
) -> str | None:
    ...
```

This is an ask/tell optimizer model embedded in a template method. It is broad enough for:

- single-candidate Markov chains;
- generational population algorithms;
- simulated annealing;
- basin hopping;
- particle-swarm methods;
- Bayesian or surrogate-assisted searches;
- future algorithms that vary population size.

The base class does not need to understand acceptance probabilities, elitism, crossover, or tournament selection.

### 3.2 Introduce explicit data contracts

Loose tuples, parallel lists, and callback-specific dictionaries should be replaced internally with dataclasses.

#### `Candidate`

```python
@dataclass
class Candidate:
    candidate_id: str
    manipulator: GBManipulator
    atom_positions: np.ndarray
    lineage: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

The candidate contains the exact manipulator associated with its coordinates. This avoids relying on a separate parallel list whose ordering must remain synchronized.

#### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    candidate_id: str
    objective: float
    evaluated_structure: Any
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    artifact: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

Important distinctions:

- `objective` replaces the energy-specific name.
- `evaluated_structure` is the structure that should become a future parent.
- `artifact` is optional output for users or provenance, not necessarily the optimizer's only structure representation.
- Failure is explicit rather than encoded only through `1.0e30`.

During the compatibility phase, `evaluated_structure` may still be a LAMMPS filename. Later interface work can supply an in-memory structure or a richer persisted structure reference without changing the algorithms.

#### `MinimizationResult`

```python
@dataclass(frozen=True)
class MinimizationResult:
    best_objective: float
    best_evaluation: EvaluationResult
    termination_reason: str
    iterations: int
    evaluations: int
    history: tuple[EvaluationRecord, ...]
```

This gives all minimizers the same public return type.

Legacy wrappers can extract the old values:

```python
def run_GA(...):
    result = self.run(...)
    return result.best_objective, str(result.best_evaluation.artifact)

def run_MC(...):
    return self.run(...).best_objective
```

#### `RunContext`

A separate mutable run context should hold:

- run ID;
- RNG;
- iteration count;
- objective-evaluation count;
- current best result;
- chronological history;
- termination reason;
- run-specific metadata.

This prevents results from separate `run()` calls from being accidentally combined.

### 3.3 Put evaluator normalization behind one collaborator

The minimizer should receive an evaluator object rather than independently managing two callback attributes.

A minimal protocol is:

```python
class ObjectiveEvaluator(Protocol):
    def evaluate(
        self,
        candidate: Candidate,
        context: EvaluationContext,
    ) -> EvaluationResult:
        ...

    def evaluate_many(
        self,
        candidates: Sequence[Candidate],
        context: EvaluationContext,
    ) -> Sequence[EvaluationResult]:
        ...
```

`evaluate_many()` may default to repeated calls to `evaluate()`. A batch-capable evaluator overrides it.

The initial candidate, Monte Carlo trials, and genetic-algorithm populations should all pass through this same interface. This removes the current inconsistency where the GA uses the single callback for initial evaluation and the batch callback only afterward.

A `CallbackEvaluator` adapter can preserve current user code:

```python
CallbackEvaluator(
    single=gb_energy_func,
    batch=gb_batch_energy_func,
)
```

It would translate:

```python
(objective, dump_filename)
```

and existing batch dictionaries into `EvaluationResult` objects.

The adapter, rather than each optimizer, should validate:

- result count;
- candidate/result identity;
- finite scalar objectives;
- output structure availability;
- serial versus batch ordering;
- callback exceptions.

### 3.4 Isolate structure reconstruction

The base class should not directly instantiate `GBManipulator` from arbitrary filenames at multiple points. Introduce one internal factory or adapter:

```python
class ManipulatorFactory(Protocol):
    def from_initial(self, structure: Any | None) -> GBManipulator:
        ...

    def from_evaluation(self, result: EvaluationResult) -> GBManipulator:
        ...

    def from_parents(
        self,
        first: EvaluationResult,
        second: EvaluationResult,
    ) -> GBManipulator:
        ...
```

The default implementation can retain all current construction requirements:

- authoritative `GBMaker`;
- `unit_cell`;
- `gb_thickness`;
- type mapping;
- current file readers.

This is particularly important while the interface corrections are incomplete. The optimizer base class should depend on a single reconstruction contract, not on temporary assumptions about how grain identity, interface location, or metadata are recovered from files.

When the ongoing interface work provides a better structure handoff, only this adapter and the evaluator result representation need to change.

### 3.5 Keep manipulation as composition

The base minimizer should not prescribe how an optimizer chooses or parameterizes structural operations.

For the first implementation, retain `Mutator` as a compatibility collaborator, but rename or wrap it as something like `LegacyMutationPool`. The optimizer receives an object with an explicit contract:

```python
class CandidateOperator(Protocol):
    def mutate(
        self,
        parent: GBManipulator,
        *,
        rng: np.random.Generator,
        context: GBMaker,
    ) -> MutationResult:
        ...

    def crossover(
        self,
        parents: Sequence[GBManipulator],
        *,
        rng: np.random.Generator,
        context: GBMaker,
    ) -> MutationResult:
        ...
```

Not every implementation needs both methods. Capability checks can be explicit rather than silently filtering unknown method names.

This keeps the `GBMinimizer` abstraction compatible with the separate `GBManipulator` abstraction project. The minimizer knows that it can request candidates; it does not encode the signatures of every manipulator operation.

### 3.6 Separate algorithm state from shared run state

#### Monte Carlo state

A private dataclass could hold:

- current accepted evaluation;
- effective temperature;
- consecutive rejection count;
- accepted step indices;
- acceptance history;
- energy-tolerance state.

Its `_ask()` produces exactly one mutation of the current accepted structure.

Its `_tell()` applies the Metropolis rule and updates the accepted state.

Its `_termination_reason()` handles maximum steps, rejection limit, energy tolerance, and any temperature-related criterion.

#### Genetic algorithm state

A separate dataclass could hold:

- current population;
- generation number;
- population lineages;
- selection parameters;
- current mating pool;
- failed-generation recovery state.

Its `_ask()` produces the next population.

Its `_tell()` performs ranking, elitist retention, mating-pool selection, and breeding-state updates.

Its `_termination_reason()` handles generation and evaluation budgets and future convergence criteria.

Neither subclass should implement evaluator dispatch, file validation, candidate identifiers, global-best bookkeeping, or generic history recording.

### 3.7 Normalize failure policy

The present behavior is inconsistent: the GA converts many failures to a penalty, while Monte Carlo generally propagates them.

The base should accept an explicit policy, for example:

```python
FailurePolicy.RAISE
FailurePolicy.PENALIZE
FailurePolicy.SKIP
```

A numerical penalty may still be provided where an algorithm requires a sortable objective, but the result must retain `status=FAILED`. This prevents a failed calculator job from being indistinguishable from a legitimate but very poor structure.

The default can preserve each legacy subclass's historical behavior during migration, while the new unified API should document one consistent default.

---

## 4. Proposed package layout

A lower-case package avoids conflicting with the existing `GBOpt/GBMinimizer.py` module:

```text
GBOpt/
    minimizers/
        __init__.py
        base.py
        models.py
        evaluation.py
        manipulation.py
        monte_carlo.py
        genetic.py
    GBMinimizer.py
```

`GBOpt/GBMinimizer.py` becomes a compatibility facade:

```python
from GBOpt.minimizers import (
    GBMinimizer,
    MonteCarloMinimizer,
    GeneticAlgorithmMinimizer,
    MinimizationResult,
)
```

This preserves imports such as:

```python
from GBOpt.GBMinimizer import GeneticAlgorithmMinimizer
```

and the existing example pattern:

```python
from GBOpt import GBMinimizer
GBMinimizer.GeneticAlgorithmMinimizer(...)
```

New code could use:

```python
from GBOpt.minimizers import GeneticAlgorithmMinimizer
```

---

## 5. Backward-compatible constructor strategy

The new preferred constructor should emphasize collaborators:

```python
GeneticAlgorithmMinimizer(
    gb=gb,
    evaluator=evaluator,
    operators=operators,
    seed=0,
    population_size=20,
    generations=50,
)
```

For a deprecation period, the concrete classes should continue to accept:

```python
GeneticAlgorithmMinimizer(
    GB,
    gb_energy_func,
    choices,
    gb_batch_energy_func=...,
)
```

The legacy arguments are converted internally into:

- `CallbackEvaluator`;
- `LegacyMutationPool`;
- default `GBManipulatorFactory`.

This avoids forcing all examples and downstream users to migrate simultaneously.

---

## 6. Incremental implementation plan

### Phase 1: Characterize current behavior

Before refactoring, add tests for behavior that is currently untested or ambiguous:

- Monte Carlo initial evaluation and acceptance.
- Per-call run IDs.
- Repeated runs and state isolation.
- Serial and batch GA equivalence.
- Callback failure behavior.
- Initial evaluated structure becoming the next parent.
- Objective names not being restricted to energy.
- Unknown or unavailable mutation choices.
- Deterministic runs under a fixed seed.

This phase should use synthetic evaluators and small structures. It should not depend on the unresolved production interface defects.

### Phase 2: Add internal data models and adapters

Introduce:

- `Candidate`;
- `EvaluationResult`;
- `EvaluationRecord`;
- `MinimizationResult`;
- `RunContext`;
- `CallbackEvaluator`;
- default manipulator factory.

At this stage, the concrete minimizers may continue to own their loops while using the new result and reconstruction contracts.

### Phase 3: Add the abstract base class

Implement the shared `run()` lifecycle and abstract ask/tell hooks.

Add a small test-only minimizer subclass to verify that a third-party optimizer can be implemented without modifying the base class.

### Phase 4: Migrate Monte Carlo

Move the existing algorithm into Monte Carlo-specific state and hooks.

Preserve `run_MC()` as a wrapper.

This migration should clarify whether mutations occur from the evaluated initial structure or the unevaluated construction structure. The architecture strongly favors the evaluated structure, but that behavior change should be explicit and tested.

### Phase 5: Migrate the genetic algorithm

Move population generation, selection, carryover, crossover, and recovery into genetic-algorithm-specific hooks.

Preserve `run_GA()` as a wrapper.

The initial candidate and every generation should use the unified evaluator dispatch.

### Phase 6: Public API and documentation

Add:

- exports from `GBOpt.minimizers`;
- a documented example of a minimal custom minimizer;
- a documented custom evaluator;
- migration guidance for old callbacks;
- warnings for deprecated constructor and run-method forms.

Only after this phase should broader features such as checkpointing, asynchronous evaluation, or persistent campaign databases be considered.

---

## 7. Why this approach was selected

The selected design most closely matches the architecture claimed by the paper while correcting the specific coupling found in the implementation.

It preserves the intended composition:

```text
GBMaker context
      |
GBMinimizer algorithm
      |
GBManipulator / candidate operators
      |
Objective evaluator
```

The base class formalizes orchestration, while concrete subclasses supply selection or acceptance policy. Serial versus batch execution becomes an evaluator concern rather than a genetic-algorithm-only feature. Structure serialization becomes an adapter concern rather than being embedded throughout each algorithm.

The ask/tell hooks are also at the correct abstraction level. An optimizer fundamentally needs to:

- request candidate evaluations;
- receive objective results;
- update its state.

This model supports both current algorithms without forcing all future algorithms into either a one-candidate or fixed-generation structure.

---

## 8. Advantages and disadvantages

### Advantages

**Strong separation of concerns.** Algorithm code no longer performs callback adaptation, file validation, run naming, structure loading, and history formatting.

**Actual extensibility.** A new minimizer implements a small set of lifecycle hooks instead of copying either existing class.

**Backend neutrality.** Results can carry an in-memory evaluated structure, a file reference, or another structure token. LAMMPS dump files become one adapter implementation.

**Unified serial and batch behavior.** Every algorithm can use batch evaluation when appropriate, and every batch evaluator has a serial fallback.

**Consistent public results.** All minimizers return `MinimizationResult`, regardless of their internal state model.

**Run isolation.** A fresh `RunContext` prevents histories and counters from leaking between repeated runs.

**Incremental migration.** Existing callbacks, constructors, imports, and run methods can remain operational.

**Compatibility with ongoing interface work.** Grain ownership and interface metadata reconstruction remain concentrated in the manipulator factory rather than being repeated throughout optimizer code.

### Disadvantages

**More types and indirection.** Simple scripts will pass through evaluator and factory adapters rather than calling one function directly.

**A larger initial refactor.** Extracting common behavior without altering Monte Carlo or genetic-algorithm semantics requires characterization tests.

**Some hooks may need refinement.** A future asynchronous optimizer may require a more explicit pending-candidate model than the initial synchronous ask/tell loop.

**Temporary dual API.** Supporting both legacy callbacks and the new evaluator interface creates a maintenance burden during the deprecation period.

**The abstraction cannot itself repair structure metadata.** Until the interface project supplies a reliable evaluated-structure handoff, the default factory remains constrained by the current file loaders.

---

## 9. Alternatives considered and rejected

### Alternative 1: A thin abstract base class with only an abstract `run()`

Under this approach, the base class would store `GB`, the callback, the mutator, and the RNG, while each subclass retained its complete loop.

#### Why it was rejected

This would provide nominal inheritance without solving the main problems. Evaluation contracts, file reconstruction, error handling, result formats, state isolation, and history would remain duplicated and inconsistent. A third optimizer would still need to copy most of an existing implementation.

### Alternative 2: One concrete minimizer configured entirely by an algorithm strategy

This would replace subclasses with:

```python
GBMinimizer(algorithm=MonteCarloStrategy(...))
```

#### Why it was rejected

Composition is attractive, but the current request and the paper specifically identify abstract base classes as the planned user extension mechanism. More importantly, Monte Carlo and genetic algorithms maintain substantially different algorithm state. A strategy interface broad enough to support both would end up looking nearly identical to the proposed abstract ask/tell hooks, while adding another forwarding layer.

Strategy objects could still be introduced later if runtime algorithm swapping becomes valuable.

### Alternative 3: A fully generic optimizer independent of GBOpt domain objects

This would model the problem as a conventional optimizer over vectors or opaque objective inputs, leaving all structure manipulation outside the optimizer framework.

#### Why it was rejected

GBOpt candidates are not ordinary numerical vectors. The optimization loop needs evaluated structures, parent relationships, manipulation lineage, structural failures, and often relaxed geometries from an external calculator. Removing all domain knowledge would either discard essential provenance or force users to rebuild GBOpt's orchestration around every generic optimizer.

A database-backed task graph or campaign engine was also considered excessive for this phase. It would solve persistence and distributed scheduling, but it is not required to establish the abstraction boundary and would couple this work to much larger workflow decisions.

---

## 10. Completion criteria

The abstraction should be considered successful when:

1. `GBMinimizer` is genuinely abstract and owns the shared run lifecycle.
2. Monte Carlo and genetic algorithm implementations subclass it without duplicating evaluator or reconstruction machinery.
3. A minimal third optimizer can be written using only the documented hooks.
4. Both serial and batch evaluators return the same normalized result type.
5. Neither concrete optimizer requires a LAMMPS-specific result field.
6. Every `run()` returns an isolated `MinimizationResult`.
7. Existing `run_MC()` and `run_GA()` callers continue to work during migration.
8. Fixed-seed behavior is reproducible.
9. Failed evaluations remain distinguishable from valid high-objective candidates.
10. Current interface reconstruction assumptions exist in one adapter rather than throughout the optimizer algorithms.

The central boundary should be:

> **The base minimizer controls execution mechanics; subclasses control optimization policy; manipulators control structural changes; evaluators control objective computation; adapters control representation handoff.**

---

## Source basis

This proposal is based on:

- the uploaded GBOpt source archive;
- French, J. C. and Bhave, C. V. (2026), *GBOpt: Grain boundary structure optimization using Monte Carlo and evolutionary algorithms*, SoftwareX 35, 102763.
