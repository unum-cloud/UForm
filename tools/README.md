# UForm Tools

Three binaries train, score and grow UForm checkpoints, each through the library the C interface wraps.
They build with `-D UFORM_BUILD_TOOLS=ON` beside `-D UFORM_BUILD_TRAIN=ON`, and land at the build root beside the suites.

- `uform_base --tier <name>` trains a tier on a `--mixture`, a mixture JSON or the synthetic `tokens` or `omni` draw, under the `--objective` it names, and writes `--checkpoint` on cadence and at the end.
- `uform_eval <checkpoint>` prints Recall@1, @5 and @10 in both directions over `--coco`, `--msrvtt`, `--clotho` and `--coco-sm`, on host threads or the GPU `--gpu` names, and with `--calibrate <choices.jsonl> --output <path>` fits the choice head's temperatures and writes the checkpoint again.
- `uform_grow <in> <out> --deepen` or `--widen` doubles a checkpoint's depth or width, and keeps the output only if its embeddings reproduce the source's to a cosine of 0.9999.

## Flags

Flags are long kebab-case words whose value follows as the next argument or after `=`, as `--steps 200` or `--steps=200`.
Counts are plain integers, reals like `5e-4` are non-negative, seeds are 32-bit or `random`, and switches like `--eager` take no value.
Enumerators read through the C name tables, so `--objective` takes `masked`, `contrastive` or `joint` as `uform_objective_name` spells them.
`--help` prints every flag with its placeholder, its meaning and its variable.

Every flag reads its default from `UFORM_<FLAG>`, its name upper-cased with dashes as underscores: `UFORM_TIER=tiny` stands for `--tier tiny`, and a switch's variable takes `1`, `0`, `true` or `false`.
The command line overrides the variable.

## Exit Codes and Progress

- `0`: the run finished.
- `1`: a flag, a variable or the run failed, with one line saying why, as `--batch="0" does not parse, expected a positive count`.
- `130`: a SIGINT or SIGTERM stopped the run.

Every step of a long run prints `<phase>: <completed> of <total> <unit>` at most once a second, as `train: 120 of 1000 steps` or `coco: 2048 of 25010 captions`, and the library's own progress reports print through the same line.
The first signal stops the run at its next barrier: `uform_base` before its next step, writing the checkpoint at the steps completed, `uform_eval` between batches, writing no calibrated checkpoint, and `uform_grow` between blocks or check encodings, deleting its unverified output.
Each then prints `stopped in <phase>` and exits with 130, and a second signal ends the process at once.
A `uform_base` run across hosts keeps the default disposition, since one host leaving the step loop alone would strand its peers.
