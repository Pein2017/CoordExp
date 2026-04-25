# Full Idea: Prefix-Attached Multi-Positive Subset Training for Autoregressive V-LLM Detection

## 0. One-sentence summary

We replace ordinary fixed-order SFT with a **subset-conditioned multi-positive objective**:

> Given an image and an arbitrary prefix containing some already-emitted GT objects, train the model so that **any remaining observed GT object** is a valid next continuation, while avoiding strong pressure to stop early.

This turns autoregressive detection from:

```text
imitate one annotated sequence
```

into:

```text
from any observed subset state, continue toward covering the observed GT set
```

The objective is still fully compatible with:

- original V-LLM language modeling path;
- original embedding / lm_head;
- autoregressive decoding;
- coord tokens;
- no detection head;
- no external proposal branch.

---

# 1. Problem setting

## 1.1 Image, prompt, and object set

Let:

$$I$$

be an image, and:

$$x$$

be the detection / grounding prompt.

The true object set is:

$$O^\star = \{o_1^\star,\ldots,o_{N^\star}^\star\}$$

Each object is:

$$o_i = (d_i, b_i, a_i)$$

where:

- $d_i$: object description / class text;
- $b_i = (x_{1i}, y_{1i}, x_{2i}, y_{2i})$: bbox represented by coord tokens;
- $a_i$: optional attributes, visibility, instance information, etc.

The observed annotation is incomplete:

$$O_{\text{obs}} \subseteq O^\star$$

The unobserved true positives are:

$$O_{\text{lat}} = O^\star \setminus O_{\text{obs}}$$

The key assumption is:

$$O_{\text{obs}} \text{ provides positive evidence, not complete evidence.}$$

That is:

$$o \in O_{\text{obs}} \Rightarrow o \text{ is positive}$$

but:

$$o \notin O_{\text{obs}} \not\Rightarrow o \text{ is negative}$$

---

## 1.2 Autoregressive sequence model

The V-LLM defines:

$$p_\theta(Y \mid I,x) = \prod_{t=1}^{T} p_\theta(y_t \mid I,x,y_{<t})$$

where $Y = (y_1,\ldots,y_T)$ is the generated detection sequence.

Each object is serialized into an entry:

$$s(o) = [y^{(o)}_1,\ldots,y^{(o)}_{L_o}]$$

The exact order of fields is not essential to the proposed objective.
The important unit is the **full object entry** $s(o)$, not individual tokens.

A full annotated sequence under order $\pi$ is:

$$S_\pi(O_{\text{obs}}) = s(o_{\pi_1}) \oplus s(o_{\pi_2}) \oplus \cdots \oplus s(o_{\pi_n})$$

where:

$$n = |O_{\text{obs}}|$$

and:

$$\pi \in \mathfrak{S}_n$$

is a permutation of observed objects.

---

# 2. What ordinary SFT does wrong

Ordinary SFT chooses one annotated order:

$$\pi_{\text{ann}}$$

and optimizes:

$$\mathcal L_{\text{SFT}} = -\log p_\theta\left(S_{\pi_{\text{ann}}}(O_{\text{obs}}) \oplus \text{EOD}\mid I,x\right)$$

Equivalently:

$$\mathcal L_{\text{SFT}} = -\sum_t \log p_\theta\left(y_t^{\text{ann}}\mid I,x,y_{<t}^{\text{ann}}\right)$$

This implicitly assumes:

$$O_{\text{obs}} = O^\star$$

and:

$$\pi_{\text{ann}} \text{ is the unique correct order}$$

and:

$$\text{after } O_{\text{obs}}, \text{ the model should stop.}$$

These assumptions are false under sparse / incomplete annotations.

---

## 2.1 Single-positive SFT as a degenerate transition objective

Suppose:

$$O_{\text{obs}} = \{A,B,C,D\}$$

and the annotation order is:

$$A \rightarrow B \rightarrow C \rightarrow D$$

Ordinary SFT trains transitions like:

$$\emptyset \rightarrow A$$

$$\{A\} \rightarrow B$$

$$\{A,B\} \rightarrow C$$

$$\{A,B,C\} \rightarrow D$$

Each transition has exactly one positive next object.

However, detection is a set prediction problem.
At prefix:

$$S = \{A\}$$

the next valid object should be any of:

$$\{B,C,D\}$$

not only $B$.

---

# 3. Proposed objective: subset-conditioned multi-positive training

## 3.1 Subset state

For each image, sample a subset of observed objects:

$$S \subset O_{\text{obs}}$$

This subset is interpreted as:

```text
objects already emitted in the prefix
```

The remaining observed positives are:

$$R(S) = O_{\text{obs}} \setminus S$$

A prefix sequence is formed by serializing $S$ in some random order:

$$h_S = \operatorname{Serialize}(\rho(S))$$

where $\rho(S)$ is a random permutation of the subset $S$.

The model is conditioned on:

$$(I,x,h_S)$$

and trained to produce one of the remaining observed objects.

---

## 3.2 Full-entry candidate probability

For each remaining object $o \in R(S)$, define its full-entry log probability:

$$g_\theta(o; S) = \log p_\theta(s(o) \mid I,x,h_S)$$

Expanded autoregressively:

$$g_\theta(o; S) = \sum_{t=1}^{L_o} \log p_\theta\left(y_t^{(o)}\mid I,x,h_S,y_{<t}^{(o)}\right)$$

This is a **teacher-forced full-entry score**.

Important: $g_\theta(o; S)$ is not a token-wise independent score.
It is the score of the whole object entry.

---

## 3.3 Multi-positive next-object likelihood

The core objective is:

$$Z_\theta(S) = \sum_{o \in R(S)} \exp(g_\theta(o; S))$$

Then:

$$\boxed{\mathcal L_{\text{MP}}(S) = -\log Z_\theta(S)}$$

Equivalently:

$$\boxed{\mathcal L_{\text{MP}}(S) = -\log \sum_{o \in O_{\text{obs}}\setminus S} p_\theta(s(o)\mid I,x,h_S)}$$

Interpretation:

```text
Given prefix S, any remaining observed GT object is a valid next continuation.
```

This replaces:

$$-\log p_\theta(s(o_{\text{next}})\mid I,x,h_S)$$

with:

$$-\log \sum_{o\in R(S)} p_\theta(s(o)\mid I,x,h_S)$$

---

# 4. Why log-sum-exp is essential

Let $g_o = g_\theta(o;S)$.

Then:

$$\mathcal L_{\text{MP}}(S) = -\log\sum_{o\in R(S)}\exp(g_o)$$

The responsibility of object $o$ is:

$$r_o = \frac{\exp(g_o)}{\sum_{o'\in R(S)}\exp(g_{o'})}$$

The gradient is:

$$\nabla_\theta \mathcal L_{\text{MP}} = -\sum_{o\in R(S)} r_o \nabla_\theta g_\theta(o;S)$$

Thus the model performs soft latent binding:

$$r_o = \text{posterior responsibility that object } o \text{ explains the current continuation}$$

This converts a hard target:

```text
the next object must be A
```

into a soft set target:

```text
the next object may be any object in R(S)
```

---

## 4.1 Relation to latent object identity

Let $z$ denote the latent identity of the next object:

$$z \in R(S)$$

The desired marginal likelihood is:

$$p_\theta(\text{valid next object}\mid I,x,h_S) = \sum_{z\in R(S)} p_\theta(s(z)\mid I,x,h_S)$$

Therefore:

$$\mathcal L_{\text{MP}} = -\log \sum_z p_\theta(s(z)\mid I,x,h_S)$$

This is exactly latent-variable marginalization over next-object identity.

---

## 4.2 Why not use max?

A hard max objective would be:

$$\mathcal L_{\max} = -\max_{o\in R(S)} g_\theta(o;S)$$

This only trains the currently easiest object.

It can cause mode starvation:

```text
large / easy objects receive all gradient;
small / occluded / difficult objects receive little or no gradient.
```

The log-sum-exp objective is a soft version:

$$\log\sum_i\exp(g_i)$$

which allows all valid modes to contribute.

---

# 5. Full-entry level, not token-wise level

## 5.1 Wrong token-wise multi-positive

A tempting but wrong objective is:

$$-\log\left[p(x_1^A) + p(x_1^B) + p(x_1^C)\right]$$

then:

$$-\log\left[p(y_1^A) + p(y_1^B) + p(y_1^C)\right]$$

This allows invalid mixtures such as:

```text
x1 from A
y1 from B
x2 from C
y2 from A
```

This can produce non-object boxes.

---

## 5.2 Correct full-entry objective

The mixture should be at the object-entry level:

$$-\log\left[p(s(A)) + p(s(B)) + p(s(C))\right]$$

or at least at the full-bbox level:

$$-\log\left[p(b_A,d_A) + p(b_B,d_B) + p(b_C,d_C)\right]$$

This preserves the correlation among $x_1,y_1,x_2,y_2,d$ and avoids coordinate-wise mode mixing.

---

# 6. Subset selection strategy

The training objective is an expectation over subsets:

$$\mathcal L_{\text{subset-MP}} = \mathbb E_{S\sim q(S\mid O_{\text{obs}})}\left[\mathcal L_{\text{MP}}(S)\right]$$

where $q(S\mid O_{\text{obs}})$ is a subset sampling distribution.

A useful mixture distribution is:

$$q(S) = \alpha_0 q_{\text{empty}}(S) + \alpha_r q_{\text{random}}(S) + \alpha_{\text{loo}} q_{\text{leave-one-out}}(S)$$

with $\alpha_0+\alpha_r+\alpha_{\text{loo}}=1$.

---

## 6.1 Empty-prefix sampling

$$S=\emptyset$$

Then $R(S)=O_{\text{obs}}$.

Objective:

$$-\log \sum_{o\in O_{\text{obs}}} p_\theta(s(o)\mid I,x)$$

Purpose:

```text
train the first-object selection distribution;
avoid bias toward a fixed first object.
```

---

## 6.2 Random subset sampling

Sample $S\subset O_{\text{obs}}$ with random size and random membership.

Purpose:

```text
train arbitrary intermediate prefix states.
```

This teaches:

$$S \rightarrow S\cup\{o\},\quad o\in O_{\text{obs}}\setminus S$$

for many possible $S$.

---

## 6.3 Leave-one-out sampling

Choose one object $o_i \sim \operatorname{Uniform}(O_{\text{obs}})$.

Set $S=O_{\text{obs}}\setminus\{o_i\}$.

Then $R(S)=\{o_i\}$.

Objective:

$$-\log p_\theta(s(o_i)\mid I,x,h_S)$$

Purpose:

```text
force every observed GT object to be recoverable from a nearly complete prefix.
```

This combats easy-object dominance in the log-sum objective.

---

# 7. Stop / continue handling

## 7.1 Stop token set

Define a set of global stop tokens:

$$\mathcal E_{\text{stop}} = \{\text{EOD}, \text{EOS}, \text{final } ], \text{final } \}, \text{``no more objects''}\}$$

The exact set depends on serialization.

Define:

$$P_{\text{stop}}(S) = \sum_{e\in\mathcal E_{\text{stop}}} p_\theta(e\mid I,x,h_S)$$

---

## 7.2 If remaining observed GT exists

If $R(S)\neq\emptyset$, then stopping is wrong because at least one observed GT object has not been emitted.

Add anti-stop loss:

$$\boxed{\mathcal L_{\text{anti-stop}}(S) = -\log\left(1-P_{\text{stop}}(S)\right)}$$

This explicitly trains:

```text
if observed GT remains, do not stop.
```

---

## 7.3 If observed GT is exhausted

If $R(S)=\emptyset$, ordinary SFT would train $-\log P_{\text{stop}}(S)$.

But under missing annotations $O_{\text{obs}} \neq O^\star$, so:

```text
observed GT exhausted
```

does not imply:

```text
no more true objects exist
```

Therefore use an annotation completeness coefficient $c_I\in[0,1]$ and train:

$$\boxed{\mathcal L_{\text{stop}}(S) = c_I \cdot [-\log P_{\text{stop}}(S)]}$$

Interpretation:

- Fully audited image: $c_I\approx 1$
- Sparse / incomplete image: $c_I\approx 0$

Thus, on sparse annotations, do not strongly force the model to stop.

---

## 7.4 Object-level end vs global end

Distinguish `</object>` (object-level end) from `<EOD>` (global end).

Object-level end is reliable: "object entry is complete"

Global end is not always reliable: "the entire detection list is complete"

Therefore:

- keep normal loss on `</object>`;
- downweight / mask / condition loss on `<EOD>`.

---

# 8. Reserving probability space for unobserved positives

Multi-positive over $O_{\text{obs}}$ solves conflicts among observed positives.

It does not fully solve missing labels.

If $O^\star = \{A,B,C,D,E\}$ but $O_{\text{obs}}=\{A,B,C\}$, then $-\log[p(A)+p(B)+p(C)]$ still indirectly suppresses $D,E$, because probability mass is normalized.

To reduce this pressure, use a positive-evidence margin.

---

## 8.1 Positive-evidence margin

Instead of always minimizing:

$$-\log Z_\theta(S)$$

define a threshold:

$$\rho_I\in(0,1]$$

and use:

$$\boxed{ \mathcal L_{\text{PEM}}(S) = \max \left( 0, \log\rho_I - \log Z_\theta(S) \right) }$$

Equivalent condition:

$$\mathcal L_{\text{PEM}}(S)=0 \quad \text{if} \quad Z_\theta(S)\geq \rho_I$$

Interpretation:

```text
observed GT probability mass only needs to be sufficiently high;
it does not need to consume all probability mass.
```

For complete annotations:

$$\rho_I \approx 1$$

For sparse annotations:

$$\rho_I < 1$$

For example:

$$\rho_I=0.5$$

means:

```text
as long as observed remaining GT gets at least 0.5 total probability,
do not keep pushing it toward 1.0.
```

This leaves probability space for latent positives.

---

# 9. Balance regularization against easy-object dominance

The log-sum objective may be dominated by easy objects.

Responsibilities:

$$r_o = \frac{\exp(g_\theta(o;S))} {\sum_{o'\in R(S)}\exp(g_\theta(o';S))}$$

If:

$$r_A\approx 1$$

and all other $r_o$ are near zero, training behaves like hard max.

A weak balance regularizer can be used:

$$\mathcal L_{\text{bal}}(S)=\operatorname{KL}\left(U_{R(S)}\,\|\,r\right)$$

where $U_{R(S)}$ is uniform over remaining observed objects.

Expanded:

$$U_{R(S)}(o)=\frac{1}{|R(S)|}$$

$$\mathcal L_{\text{bal}}=\sum_{o\in R(S)}\frac{1}{|R(S)|}\log\frac{1/|R(S)|}{r_o}$$

This is optional.

A safer and often simpler alternative is leave-one-out sampling.

---

# 10. Final training loss

For a sampled subset $S$:

## Case 1: $R(S)\neq\emptyset$

Define:

$$g_o = g_\theta(o;S)$$

$$Z_\theta(S) = \sum_{o\in R(S)} \exp(g_o)$$

Base multi-positive loss:

$$\mathcal L_{\text{MP}}(S) = -\log Z_\theta(S)$$

Positive-evidence margin variant:

$$\mathcal L_{\text{PEM}}(S) = \max \left( 0, \log\rho_I - \log Z_\theta(S) \right)$$

Anti-stop:

$$\mathcal L_{\text{anti-stop}}(S) = -\log \left( 1-P_{\text{stop}}(S) \right)$$

Optional balance:

$$\mathcal L_{\text{bal}}(S) = \operatorname{KL} \left( U_{R(S)} \| r \right)$$

Recommended full loss:

$$\boxed{ \mathcal L(S) = \mathcal L_{\text{PEM}}(S) + \lambda_{\text{stop}} \mathcal L_{\text{anti-stop}}(S) + \beta \mathcal L_{\text{bal}}(S) }$$

Minimal version:

$$\boxed{ \mathcal L(S) = -\log \sum_{o\in R(S)} p_\theta(s(o)\mid I,x,h_S) + \lambda_{\text{stop}} [-\log(1-P_{\text{stop}}(S))] }$$

---

## Case 2: $R(S)=\emptyset$

Use completeness-weighted stop loss:

$$\boxed{ \mathcal L(S) = c_I [-\log P_{\text{stop}}(S)] }$$

For sparse annotations:

$$c_I \approx 0$$

For fully verified annotations:

$$c_I \approx 1$$

---

# 11. Prefix attach

## 11.1 Goal

When scoring multiple candidates under the same prefix:

$$s(o_1),s(o_2),\ldots,s(o_K)$$

we want all candidates to condition on the same prefix:

$$h_S$$

but not on each other.

Conceptual computation graph:

```text
shared prefix h_S
   ├── candidate branch s(o_1)
   ├── candidate branch s(o_2)
   ├── candidate branch s(o_3)
   └── ...
```

Each branch computes:

$$g_\theta(o_k;S) = \log p_\theta(s(o_k)\mid I,x,h_S)$$

Then:

$$\mathcal L_{\text{MP}} = -\log \sum_k \exp(g_\theta(o_k;S))$$

---

## 11.2 Why branches must not see each other

If candidates are naively concatenated:

```text
prefix + candidate_A + candidate_B + candidate_C
```

then candidate $B$ can attend to candidate $A$, and candidate $C$ can attend to both $A$ and $B$.

This corrupts the probability:

$$p(s(B)\mid I,x,h_S)$$

into:

$$p(s(B)\mid I,x,h_S,s(A))$$

which is not the intended score.

Correct semantics:

$$s(o_i) \perp s(o_j) \quad \text{given} \quad (I,x,h_S)$$

for candidate branches.

---

## 11.3 Tree / block causal mask semantics

The intended attention pattern is:

| Query tokens | Can attend to prefix | Can attend to own branch | Can attend to other branches |
|---|---:|---:|---:|
| prefix | causal prefix only | no | no |
| branch A | yes | causal branch A | no |
| branch B | yes | causal branch B | no |
| branch C | yes | causal branch C | no |

Symbolically:

$$\operatorname{Attn}(A_t) \subseteq h_S \cup A_{<t}$$

$$\operatorname{Attn}(B_t) \subseteq h_S \cup B_{<t}$$

but:

$$A_t \not\rightarrow B$$

$$B_t \not\rightarrow A$$

This preserves:

$$p_\theta(s(o_k)\mid I,x,h_S)$$

for all candidate branches.

---

## 11.4 Detach vs non-detach prefix attach

Let:

$$u_\theta(S) = H_\theta(I,x,h_S)$$

be the prefix hidden state / KV representation.

Candidate score:

$$g_\theta(o;S) = G_\theta(u_\theta(S), s(o))$$

### Detached prefix version

If prefix representation is detached:

$$u_\theta(S) \leftarrow \operatorname{stopgrad}(u_\theta(S))$$

then:

$$\nabla_\theta g_\theta(o;S) = \frac{\partial G_\theta}{\partial \theta}$$

The gradient does not flow through:

$$u_\theta(S)$$

This trains:

```text
candidate continuation given fixed prefix representation
```

but not:

```text
how prefix representation should change to support better continuation
```

### Non-detached prefix version

If prefix is not detached:

$$\nabla_\theta g_\theta(o;S) = \frac{\partial G_\theta}{\partial \theta} + \frac{\partial G_\theta}{\partial u_\theta} \frac{\partial u_\theta}{\partial \theta}$$

This allows candidate losses to update the shared prefix representation.

This is closer to ordinary SFT, where earlier token states participate in the computation graph.

---

## 11.5 Relation to SFT

Standard SFT is non-detached:

$$\mathcal L_{\text{SFT}} = -\sum_t \log p_\theta(y_t\mid y_{<t},I,x)$$

Every hidden state depends on $\theta$.
Loss at later tokens can backpropagate through the computation graph involving earlier token representations.

Therefore:

```text
ordinary SFT ≈ non-detached full-sequence training
```

The proposed ideal version is:

```text
non-detached shared-prefix multi-branch training
```

The detached-cache version is only an approximation.

---

# 12. Complexity

Let:

- $n = |O_{\text{obs}}|$: number of observed GT objects;
- $s = |S|$: number of objects in prefix;
- $r = |R(S)| = n-s$: number of remaining objects;
- $L$: average tokens per object entry.

## 12.1 Ordinary SFT

Full sequence cost in entry-token units:

$$C_{\text{SFT}} = nL$$

---

## 12.2 Naive replicated multi-positive

If each candidate repeats the full prefix:

$$C_{\text{naive}} = r(sL+L) = r(s+1)L$$

This can be much larger than SFT.

Example:

$$n=30,\quad s=20,\quad r=10,\quad L=20$$

$$C_{\text{naive}} = 10(20+1)20 = 4200$$

while:

$$C_{\text{SFT}} = 30\cdot20 = 600$$

---

## 12.3 Prefix-attached multi-positive

If prefix is shared once and branches are attached:

$$C_{\text{attach}} = sL + rL = nL$$

Thus, if all remaining objects are scored:

$$C_{\text{attach}} \approx C_{\text{SFT}}$$

in token-budget terms.

For:

$$n=30,\quad s=20,\quad r=10,\quad L=20$$

$$C_{\text{attach}} = 20\cdot20 + 10\cdot20 = 600$$

same as ordinary SFT token count.

---

## 12.4 Candidate subsampling

If remaining set is large, sample candidates:

$$C(S)\subset R(S)$$

with:

$$|C(S)|=K$$

Then:

$$Z_\theta(S) \approx \widehat Z_\theta(S) = \frac{|R(S)|}{K} \sum_{o\in C(S)} \exp(g_\theta(o;S))$$

Loss:

$$\widehat{\mathcal L}_{\text{MP}} = -\log \widehat Z_\theta(S)$$

The multiplicative factor:

$$\frac{|R(S)|}{K}$$

is constant with respect to $\theta$ for fixed $S,C(S)$, so it does not change the gradient direction under plain $-\log\widehat Z$.
It matters for calibrated thresholds such as positive-evidence margin.

---

# 13. Same-budget benchmark design

The goal is to compare ordinary SFT and proposed subset-MP under similar compute.

---

## 13.1 Baseline A: ordinary SFT

Random order full sequence:

$$S_{\pi}(O_{\text{obs}})$$

Loss:

$$-\log p_\theta(S_\pi(O_{\text{obs}})\oplus \text{EOD}\mid I,x)$$

---

## 13.2 Baseline B: SFT without global EOD

Same as ordinary SFT, but mask or downweight final EOD:

$$\mathcal L = -\sum_{t\in\text{object tokens}} \log p_\theta(y_t\mid I,x,y_{<t})$$

No strong global stop training.

Purpose:

```text
is conservative stopping the main bottleneck?
```

---

## 13.3 Proposed C: one-prefix exact multi-positive

Sample one subset $S$.

Score all remaining objects:

$$R(S)=O_{\text{obs}}\setminus S$$

Loss:

$$\mathcal L = -\log \sum_{o\in R(S)} p_\theta(s(o)\mid I,x,h_S) + \lambda_{\text{stop}} [-\log(1-P_{\text{stop}}(S))]$$

If prefix attach is used, token budget:

$$sL+(n-s)L=nL$$

approximately equal to ordinary SFT.

Purpose:

```text
test whether multi-positive set-state objective beats fixed-order sequence imitation under the same token budget.
```

---

## 13.4 Proposed D: nested multi-prefix budgeted objective

Choose a random order:

$$\pi = (o_{\pi_1},\ldots,o_{\pi_n})$$

Choose checkpoints:

$$0=k_1 < k_2 < \cdots < k_m < n$$

Prefix at checkpoint $j$:

$$S_j = \{o_{\pi_1},\ldots,o_{\pi_{k_j}}\}$$

Remaining:

$$R_j = O_{\text{obs}}\setminus S_j$$

Sample candidate subset:

$$C_j\subset R_j$$

Loss:

$$\mathcal L = \frac{1}{m} \sum_{j=1}^{m} \left[ -\log \sum_{o\in C_j} p_\theta(s(o)\mid I,x,h_{S_j}) + \lambda_{\text{stop}} [-\log(1-P_{\text{stop}}(S_j))] \right]$$

Budget condition:

$$k_{\max}+\sum_j |C_j| \approx n$$

where:

$$k_{\max} = \max_j k_j$$

Example for $n=30$:

```text
checkpoints: [0, 7, 15, 22]
candidate counts: [2, 2, 2, 2]
```

Budget:

$$22 + 2+2+2+2 = 30$$

which matches ordinary SFT entry-token budget.

Purpose:

```text
train multiple prefix states in one image while preserving approximately the same token budget.
```

---

# 14. Relationship to full permutation marginalization

The ideal observed-set marginal likelihood is:

$$\mathcal L_{\text{perm}} = -\log \sum_{\pi\in\mathfrak S_n} p_\theta(S_\pi(O_{\text{obs}})\mid I,x)$$

This marginalizes over all object orders.

A dynamic-programming view:

$$F(S) = \log \sum_{o\in O_{\text{obs}}\setminus S} \exp \left[ g_\theta(o;S) + F(S\cup\{o\}) \right]$$

with terminal condition:

$$F(O_{\text{obs}})=0$$

if no EOD is forced.

Then:

$$-F(\emptyset)$$

is the exact permutation-marginalized loss over observed objects.

However, exact DP requires:

$$2^n$$

subset states and is infeasible for large $n$.

The proposed subset-MP objective is a Monte Carlo local approximation:

$$\mathbb E_{S\sim q(S)} \left[ -\log \sum_{o\in O_{\text{obs}}\setminus S} p_\theta(s(o)\mid I,x,h_S) \right]$$

It trains local transitions:

$$S \rightarrow S\cup\{o\}$$

instead of enumerating complete paths.

---

# 15. Relationship to incomplete annotations

The true ideal positive-evidence objective is not merely permutation marginalization over $O_{\text{obs}}$.

It is:

$$\boxed{ -\log \sum_{Y:\ D(Y)\supseteq O_{\text{obs}}} p_\theta(Y\mid I,x) }$$

where:

$$D(Y)$$

is the object set decoded from generated sequence $Y$.

Meaning:

```text
any sequence is valid if it contains all observed GT objects,
even if it also contains additional unobserved true objects.
```

This is the ideal objective for sparse annotations.

It is generally intractable because:

- unobserved objects are unknown;
- possible bboxes are huge;
- descriptions are open-vocabulary;
- sequence length is unbounded;
- termination is uncertain.

The proposed method approximates this ideal by:

1. multi-positive over observed remaining objects;
2. anti-stop when observed objects remain;
3. weak or masked stop loss when observed objects are exhausted;
4. optional positive-evidence margin to avoid forcing observed probability mass to 1;
5. later stage-2 pseudo-labeling / rollout can expand the positive set.

---

# 16. Degenerate cases

## 16.1 Ordinary SFT as a special case

If:

$$S=\{o_{\pi_1},\ldots,o_{\pi_{k-1}}\}$$

and:

$$R(S)=\{o_{\pi_k}\}$$

then:

$$\mathcal L_{\text{MP}} = -\log p_\theta(s(o_{\pi_k})\mid I,x,h_S)$$

This is ordinary single-positive teacher forcing for that transition.

Thus, SFT is a degenerate case of subset-MP with singleton remaining target.

---

## 16.2 Full observed set classification

If:

$$S=\emptyset$$

then:

$$\mathcal L_{\text{MP}} = -\log \sum_{o\in O_{\text{obs}}} p_\theta(s(o)\mid I,x)$$

This trains the first generated object to be any observed object.

---

## 16.3 Leave-one-out coverage

If:

$$S=O_{\text{obs}}\setminus\{o_i\}$$

then:

$$\mathcal L_{\text{MP}} = -\log p_\theta(s(o_i)\mid I,x,h_S)$$

This forces object $o_i$ to remain recoverable even after all other observed objects are already in the prefix.

---

# 17. Recommended minimal objective

For a first implementation / benchmark, use:

$$\boxed{ \mathcal L(S) = -\log \sum_{o\in R(S)} p_\theta(s(o)\mid I,x,h_S) + \lambda_{\text{stop}} [-\log(1-P_{\text{stop}}(S))] }$$

when:

$$R(S)\neq\emptyset$$

and:

$$\boxed{ \mathcal L(S) = c_I[-\log P_{\text{stop}}(S)] }$$

when:

$$R(S)=\emptyset$$

Recommended initial choices:

```text
lambda_stop: small positive value, e.g. 0.05 ~ 0.2
c_I: 0 for sparse labels, 1 for fully audited data
balance regularizer: off initially
positive-evidence margin: off initially, then add later
```

---

# 18. Recommended subset distribution for first benchmark

For each image:

```text
30% empty prefix
50% random subset prefix
20% leave-one-out prefix
```

More formally:

$$q(S) = 0.3q_{\text{empty}} + 0.5q_{\text{random}} + 0.2q_{\text{leave-one-out}}$$

For $q_{\text{random}}$:

- sample subset size $s$ uniformly from:

$$1,\ldots,n-2$$

if $n\geq 3$;

- sample $S$ uniformly among subsets of size $s$;
- serialize $S$ in random order.

---

# 19. Recommended same-budget benchmark

## Group A: ordinary SFT

```text
fixed/random full order
object token CE
normal final EOD CE
```

## Group B: SFT without final EOD

```text
same as SFT
mask or downweight final EOD
```

## Group C: one-prefix exact MP

```text
sample one prefix S
score all remaining R(S)
same approximate token budget as SFT if prefix attach is used
```

## Group D: nested multi-prefix budgeted MP

```text
sample several nested prefix checkpoints
score small candidate sets per checkpoint
choose budget so k_max + sum(K_j) ≈ n
```

---

# 20. Mechanism metrics

Do not only evaluate mAP.

Track the following.

## 20.1 Stop probability

$$P_{\text{stop}}(S)$$

Measure:

```text
Does the model stop too early under partial prefixes?
```

---

## 20.2 Continuation probability

$$P_{\text{cont}}(S) = 1-P_{\text{stop}}(S)$$

Measure:

```text
Does the model preserve probability of continuing?
```

---

## 20.3 Remaining-GT mass

$$Z_\theta(S) = \sum_{o\in R(S)} p_\theta(s(o)\mid I,x,h_S)$$

Measure:

```text
How much probability mass is assigned to valid remaining observed objects?
```

---

## 20.4 Responsibility entropy

$$r_o= \frac{\exp(g_o)} {\sum_{o'}\exp(g_{o'})}$$

$$H(r) = -\sum_o r_o\log r_o$$

Low entropy means:

```text
one easy object dominates the log-sum
```

High entropy means:

```text
multiple remaining objects are plausible under the prefix
```

---

## 20.5 Prefix sensitivity

For two different prefixes $S_1,S_2$ with the same remaining object set size, compare:

$$D_{\text{KL}} \left( r(\cdot\mid S_1) \| r(\cdot\mid S_2) \right)$$

or compare next-object distributions.

Purpose:

```text
Does the model depend too strongly on arbitrary prefix order?
```

---

## 20.6 Duplicate rate

After decoding:

$$\operatorname{DuplicateRate} = \frac{ \#\{(\hat b_i,\hat b_j): \operatorname{IoU}(\hat b_i,\hat b_j)>\tau,\ \hat d_i=\hat d_j\} }{ \#\{\hat b_i\} }$$

Purpose:

```text
Does subset-MP reduce or increase repeated generation?
```

---

## 20.7 Hidden-object recall

On a small fully audited subset, evaluate whether the model recovers true objects missing from the original sparse labels.

Purpose:

```text
Does anti-stop / EOD censoring actually help discover latent positives?
```

---

# 21. Expected outcomes

If the theory is correct:

1. SFT without final EOD should reduce conservative stopping.
2. One-prefix exact MP should improve observed GT recall under arbitrary order.
3. Leave-one-out sampling should improve long-tail / small / occluded object coverage.
4. Responsibility entropy should be healthier than hard single-target SFT.
5. Prefix sensitivity should decrease.
6. Continuation probability should increase under partial prefixes.
7. Sparse-eval FP may increase, but fully audited hidden-object recall should improve.

---

# 22. Core thesis

The proposed paradigm replaces:

$$\text{sequence imitation}$$

with:

$$\text{subset-conditioned set continuation}$$

Ordinary SFT says:

```text
Given this prefix, the next object is exactly the annotated next object.
```

Subset-MP says:

```text
Given this prefix, any remaining observed object is a valid next object.
```

EOD censoring says:

```text
Observed GT exhausted does not necessarily mean the world is exhausted.
```

Prefix attach says:

```text
Multiple candidate continuations should share the same prefix state,
but remain independent branches,
and ideally all candidate losses should backpropagate into the shared prefix representation.
```

The method is not a complete solution to missing labels, because unknown positives are not explicitly included in the positive set.

However, it removes several major wrong assumptions of ordinary SFT:

1. fixed object order;
2. singleton next-object target;
3. observed GT as complete world;
4. final EOD as reliable supervision;
5. coordinate / bbox mode treated as one-hot even under object ambiguity.

The practical stage-1 recipe is:

```text
subset selection
+ full-entry multi-positive log-sum-exp
+ prefix attach
+ anti-stop when remaining GT exists
+ weak/masked stop when annotation completeness is low
```
