# External data and implementation audit

This note records which external sources informed Soma, what was adopted, and
what was deliberately not used as training supervision. Human-looking output is
not evidence of human provenance; every source keeps an explicit role.

## Pointer sources

### Trusted Soma capture

The privacy-safe browser capture is the positive sequence corpus. Flow
preparation retains every finite, temporally measurable target-directed movement
regardless of distance, duration, straightness, curvature, hesitation, or
correction count. Session IDs define group-disjoint validation cohorts. A record
with zero start-to-end displacement remains valid discriminator evidence but has
no target axis, so it cannot enter the target-conditioned flow representation.

### `featurized_mouse_data.csv`

The local file is byte-identical to Figshare dataset version 2
(`15a37020a2b88afa8e5e7980010761c3`, MD5; DOI
`10.6084/m9.figshare.29386898.v2`). It contains 157,374 rows of 38 aggregate
features. Its labels describe normal versus anomalous/unauthorized access, not
human versus generated trajectories, and it contains no point sequence.

It is unsafe as positive flow supervision: 109,042 rows are duplicate feature
rows, and 108,598 rows belong to identical-feature groups containing conflicting
labels. It may be used only as a declared aggregate anomaly benchmark after
duplicate-group isolation.

### DELBOT-Mouse

Useful idea: preserve local temporal order and expose velocity, acceleration,
angle, and jerk rather than relying only on aggregate statistics. Soma applies
that lesson to the flow conditioner by retaining phase instead of globally
averaging the sequence. The published chunk-level random split can place windows
from one source session on both sides of an evaluation, so Soma does not copy
that split strategy.

Source: <https://github.com/chrisgdt/DELBOT-Mouse>

### BeCAPTCHA-Mouse

Useful idea: evaluate the Cartesian product of three path-function families and
three velocity profiles, and measure neuromotor Sigma-Lognormal structure. Soma
uses independent implementations of the nine classes as named hard negatives
and a continuous Sigma-Lognormal audit metric. The repository is a benchmark of
generated families, not timestamped browser-human positive supervision.

Sources: <https://github.com/BiDAlab/BeCAPTCHA-Mouse> and
<https://arxiv.org/abs/2005.00890>

### WindMouse

Useful as a hard negative: attraction, persistent random force, damping,
velocity clipping, integer quantization, and fixed cadence produce a recognizable
low-order controller family. Soma implements that behavior independently and
does not copy GPL source.

Source: <https://github.com/AsfhtgkDavid/windmouse>

## Keystroke sources

### Trusted Soma capture

Soma now models key-up-to-next-key-down timing directly. Negative intervals are
real rollover, not invalid data. In the current trusted capture, 38.45% of
printable transitions overlap; overlap differs materially by digraph class
(cross-hand 48.45%, same-hand 17.10%, after-punctuation 33.49%, after-space
42.62%). The synthesizer samples this signed interval and derives keydown-to-
keydown timing from it.

### keystroke-dynamics-datagen

Useful vocabulary: hold time, keydown-to-keydown latency, and keyup-to-next-
keydown latency should all be retained. The repository records a repeated fixed
password and keeps one active timestamp per key label, so repeated/overlapping
same-key events can overwrite state. It is therefore a reference for feature
definitions, not training data or a recorder implementation.

Source: <https://github.com/nileshprasad137/keystroke-dynamics-datagen>

### user-keystroke-classification and Aalto 136M

Useful idea: robust per-digraph medians and broad multi-user coverage. The Aalto
corpus contains 136 million keystrokes from 168,000 volunteers and explicitly
documents rollover. The downstream repository's pair-construction function
names/labels appear reversed (same-user pairs are called negative and different-
user pairs positive), and its random splitting is unsuitable for identity-
disjoint claims. Soma therefore adopts the rollover insight but does not ingest
its derived labels.

Sources: <https://github.com/trabbani/user-keystroke-classification> and
<https://userinterfaces.aalto.fi/136Mkeystrokes/>

## Evaluation rules

- Human capture sessions never cross train/validation/test boundaries.
- Soma-generated paths never become human-labeled training examples.
- Generator implementations are negative families, not substitutes for human
  diversity.
- Metrics are reported by negative source so aggregate accuracy cannot conceal a
  missed family.
- The current human corpus is still one-operator data; no result establishes
  population-level fidelity.
