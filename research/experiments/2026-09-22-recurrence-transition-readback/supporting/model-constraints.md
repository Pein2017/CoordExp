# Stationary recurrence constraint and positional countermodel

Status: **candidate adviser derivation**, 2026-09-22; lead owns acceptance.
Scope: [the current unit](../unit.md), CPU-only mathematics; no trace findings,
model execution, or intervention authorization is asserted here.

## Necessary prediction

Assume each identical complete-row action applies exactly one update
`q[n+1] = lambda*q[n] + (1-lambda)`,
`a[n+1] = rho*a[n] + eta`, and a fixed-pair margin is exactly
`d[n] = b + g*q[n] - a[n]`, with stationary parameters and
`0 < lambda,rho < 1`. Then

```text
a_inf = eta/(1-rho)
d[n] = d_inf + A*lambda**n + B*rho**n
d_inf = b + g - a_inf; A = g*(q[0]-1); B = a_inf-a[0]
Delta[n] = d[n+1]-d[n] = C*lambda**n + D*rho**n
C = A*(lambda-1); D = B*(rho-1)
```

For distinct rates, divide by positive `rho**n`: the sign is that of
`C*(lambda/rho)**n + D`, a monotone sequence (or a constant). Thus nonzero
first differences have **at most one sign reversal**. An isolated zero does
not count as a reversal. Equal rates collapse to `(C+D)*lambda**n`; zero
coefficients give a single exponential or a constant. There is no
`n*lambda**n` term: these recurrences are uncoupled. Nearly equal rates weaken
numerical identifiability, not this exact bound. Negative/complex rates,
coupling, additional states, nonlinear readout, or time-varying parameters
are different models. Under the intended extra assumptions `g >= 0`,
`q[0] <= 1`, `a[0] <= a_inf`, and `lambda < rho`, any reversal is specifically
positive-to-negative; a negative-to-positive reversal already contradicts
that narrower fast-positive/slow-negative model.

## What a trace can test

- Use one fixed repeat token `r` and one fixed competitor `c`, at the same
  serialized row offset and identical within-row prefix, throughout one
  contiguous exact-repeat run. Measure `log p(r)-log p(c) = z(r)-z(c)`.
  This requires the explicit modeling assumption that this observable is
  `d`, or a fixed affine transform of it. It does not follow merely from
  calling `q` and `a` hidden evidence states.
- A repeat-versus-best-other margin is an envelope when competitor identity
  changes. The unsigned winner/runner-up gap also folds the sign at a fork.
  Neither is automatically the fixed-pair observable. Top-two logs suffice
  for exact values only when both named tokens are retained. Bounds remain
  bounds; absent values cannot be filled using another token's gap.
- Four consecutive exact margins can suffice: certified slopes `+,-,+` or
  `-,+,-` reject the bound. If each margin lies in `[L[n],U[n]]`, certify
  positive only when `L[n+1] > U[n]`, negative only when
  `U[n+1] < L[n]`. Unknown slopes do not acquire signs. With per-margin
  absolute error `epsilon`, require slope magnitude greater than
  `2*epsilon`; derive epsilon from retained precision/error evidence rather
  than a convenient post hoc threshold. This is a sufficient rejection
  rule, not a parameter-fitting procedure or a test of all possible bounds.
- The exit row is comparable through its first changed token; later
  positions have different within-row conditioning. After the differing
  row, the identical-action recurrence no longer applies. Different roles,
  episodes, or row likelihoods must not be pooled into one scalar trajectory.
  Invalid boxes can still test literal-token dynamics, but not an
  exclusion process defined over valid physical objects without another
  mapping assumption. Growing position, prefix length, and contextual K/V
  state can violate stationarity despite identical emitted rows.

## Explicit alternative without adaptive exclusion

Consider two attention heads attending only fixed anchor/null keys, with
values `+1/-1`. A rotating query against a fixed key can give score
difference `cos(theta*n)`; its weighted value is
`2*sigmoid(cos(theta*n))-1 = tanh(cos(theta*n)/2)`. This follows from the
rotation identity `(R(p*w)q)^T R(j*w)k = q^T R((j-p)*w)k`; a fixed-length
row advances relative phase by a constant. This is a constructed positional
query-key mechanism, not an assertion about this checkpoint's heads or
frequencies. Define the fixed-pair decision margin

```text
d[n] = tanh(0.5*cos(pi*n/7)) + 0.2*tanh(0.5*cos(pi*n))
n=0..4:  0.554541, 0.329874, 0.394447, 0.018380, -0.018380
slopes: -0.224667, 0.064573, -0.376067, -0.036760
```

Let this token decision choose the repeated coordinate when positive and
another coordinate when negative; hold other row tokens deterministic.
Four exact rows then a numerical exit occur with two preceding slope
reversals. The heads never read emitted rows and have no adaptive exclusion
state; only position advances. Numerical exit says nothing about new-owner
recovery. A CPU arithmetic check verified these signs and the closed-form
recurrence at three parameter settings, including equal rates; the proof
above, rather than that finite check, establishes the bound.

## Smallest next discriminator, if needed

First certify fixed identities, conditioning, and precision from saved data.
If multiple slope reversals survive, the exact stationary two-state mapping
is already rejected; no follow-up is needed to reject it, and generic
adaptive history response is not thereby rejected.

For the **specified** position-only alternative, a separately authorized
paired one-step score could hold tokens and prefix K/V fixed while changing
only the current query's rotary phase relative to those keys. Predeclare
the two phases and predicted fixed-pair contrast. The constructed positional
model predicts a phase-dependent margin; the stationary toy predicts
invariance only under its assumption that this manipulation leaves `q`,
`a`, and their readout unchanged. Shifting all query/key positions together
is not this test: it preserves their relative phase. A measured response
would establish positional sensitivity, not positional sufficiency or
absence of adaptive exclusion; position can also change history retrieval.
Broad families with unspecified position/history interactions cannot be
distinguished by this pair. This remains a bounded proposal outside the
current CPU-only package, not its automatic next stage.
