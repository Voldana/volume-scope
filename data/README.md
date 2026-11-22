# Datasets

Put downloaded volumes here. This folder is git-ignored apart from this file.

VolumeScope reads headerless `.raw` files named with the Open SciVis convention:

```
<name>_<width>x<height>x<depth>_<type>.raw      e.g. bonsai_256x256x256_uint8.raw
```

Supported types: `uint8`, `uint16`, `float32` (alias `float`).

## Where to get data

- Open SciVis Datasets: https://klacansky.com/open-scivis-datasets/ (CT and MRI scans, simulations; a few MB up to many GB)
  - Small, good first tests: `bonsai`, `foot`, `engine`, `skull`
  - Large, for out-of-core streaming (stage 3): `kingsnake`, `chameleon`, `richtmyer_meshkov`

Run:

```
volumescope data/bonsai_256x256x256_uint8.raw
```
