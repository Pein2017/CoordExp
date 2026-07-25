# Research Alignment Examples

Use these examples when a research unit moves from diagnosis to treatment or
from a pilot to scale. They illustrate patterns; they are not mandatory domain
templates.

## Positive Patterns

### Local signal with an explicit transfer test

The intended outcome is reliable free-running structured output. A local loss
improves one boundary decision under a fixed input. The unit calls this a proxy,
measures the local effect first, then evaluates complete free-running outputs
and checks that semantic quality is retained. The treatment is promoted only if
the transfer survives.

Why this is aligned: the local intervention, final evaluation, transfer claim,
and preservation requirement are all explicit.

### Mechanism evidence followed by a signal-supply census

A causal patch changes a selected internal state. Before training on that
handle, the next unit counts how often the state occurs, whether comparisons are
independent and correctly attributed, and whether the data spans the intended
population. Sparse supply leads to recollection or a narrower claim, not a
different objective chosen only because its labels are convenient.

Why this is aligned: existence, prevalence, and treatment evidence remain
separate.

### End-to-end operational smoke

A distributed collector changes one request branch. The smoke covers an
ordinary request, the changed branch, an invalid input, a worst-case length,
merge, parsing, and the final evaluator receipt before the full fan-out starts.

Why this is aligned: the smoke reaches the consumer that owns the scientific
observation rather than stopping at intermediate files.

### Domain-specific example: order-independent enumeration

The intended outcome is the final set of distinct valid items, independent of
serialization order. Training one convenient next row may remain a diagnostic
or auxiliary signal, but it is not called final-set supervision unless the loss
and evaluation preserve existing items, credit newly added items, and keep
incomparable exchanges neutral.

Why this is aligned: the example distinguishes a local serialization proxy from
the decision-owning structured outcome.

## Negative Patterns And Repairs

### Proxy substitution

**Negative:** a token margin or teacher-forced row improves, so the unit claims
that complete native behavior improved.

**Repair:** name the local result, test the causal transfer under free behavior,
and keep the final claim bounded until it passes.

### Data-convenience inversion

**Negative:** the available dataset contains many examples for one local target,
so that target silently becomes the research objective. A different grouping
has few examples, so the broader hypothesis is declared unsupported.

**Repair:** classify the shortage as a signal-supply problem. Recollect, regroup,
or narrow the unit without rewriting the intended outcome.

### Aggregate cancellation

**Negative:** one mean metric is flat, although some capabilities improve and
others regress, or a mean improves while a required capability is lost.

**Repair:** report gained, retained, lost, and neutral components before the
aggregate verdict, using domain-appropriate terms.

### Scale before alignment

**Negative:** add more data, updates, seeds, reviewers, or runtime hardening
before establishing that the treatment targets the decision-owning behavior.

**Repair:** pass the alignment and signal-supply checks first; scale one remaining
uncertainty at a time.

### Producer-only smoke

**Negative:** shards exist, so a distributed run is considered healthy even
though merge, resume, parsing, or evaluation has never consumed them.

**Repair:** require one representative end-to-end receipt and a known worst-case
input before fan-out.

### Review treadmill

**Negative:** another general reviewer is added even though the artifact,
evidence, open findings, and decision have not changed.

**Repair:** close the surface after a no-new-decision review. Continue with a
probe, bounded fix, narrowed claim, user decision, or stop.
