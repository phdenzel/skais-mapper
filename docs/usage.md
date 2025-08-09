# Usage

`skais-mapper` implements a few sub-commands for generating and
manipulating simulation data.  Use the following to see what valid
sub-commands exist:

```bash
[uv run] skais-mapper -h
```

`skais-mapper` sub-commands implement the hydra configuration
management framework. For more information on sub-command usage,
inspect the `skais_mapper/configs/` directory, or use:

```bash
[uv run] skais-mapper [sub-command] -h
```

For instance, the command to generate 1000 images from snapshot 50 is
as follows:

```bash
[uv run] skais-mapper generate +experiment=tng50-1-50-2D-0000-1000
```
