# Install

Tagged releases are available as PyPI packages. To install the latest
package, run:

```bash
pip install skais-mapper
```

For the bleeding-edge package directly from a non-main git branch, use
```bash
pip install git+https://github.com/phdenzel/skais-mapper.git@<latest-branch>
```

(and replace `<latest-branch>` with the actual branch name) or clone
the repository and run the following command in the root directory of
the repository:

```bash
pip install -e .
```


### Requirements

Building from scratch thus requires `cython`, however `skais-mapper`
ships with pre-compiled C files, making the minimal requirements

- `python >= 3.10`
- `gcc` (on linux) / `clang` (on macOS)

Also see `pyproject.toml` for the relevant python packages.


## uv

`skais-mapper` is developed using [`uv`](https://docs.astral.sh/uv/)
and thus provides a `uv.lock` file which should make installing the
package easier, faster, and universal. In the project, run

```bash
uv sync [--all-groups]
```

To add `skais-mapper` to your own project simply useful
```bash
uv add skais-mapper
```
and it will appear as a dependency in your `pyproject.toml`.


## nix

If reproducibility is of utmost importance, you might want to look
into `nix`. `skais-mapper` is packaged as nix package (see
`default.nix`). To install it, you can add the following to your 
nix-config and rebuild it

```nix
{pkgs, ...}: let
	remote = builtins.fetchurl {
		url = "https://raw.githubusercontent.com/phdenzel/skais-mapper/refs/heads/main/default.nix";
		sha256 = "sha256:13wqi39qy3hm4acjpyna591jdc22q0nz710qfirahjsl8w7biiys";
	};
	skais-mapper = pkgs.callPackage remote {
		src = pkgs.fetchFromGitHub {
		owner = "phdenzel";
		repo = "skais-mapper";
		rev = "main";
		sha256 = "sha256:0232xx762a8x73lp0b6hal0aphxggwnh1hfgk6592hxyn5r1sz50";
	};
in {
	environment.systemPackages = with pkgs; [
		skais-mapper
	];
}
```
