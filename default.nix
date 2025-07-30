{
  pkgs ? import <nixpkgs> {},
  src ? ./.,
  # subdir ? "",
}:
let 
  pythonPackage = pkgs.python312Packages.buildPythonApplication {
    pname = "skais-mapper";
    version = "0.1.1";
    format = "pyproject";
    build-system = with pkgs.python312Packages; [setuptools wheel numpy];
    propagatedBuildInputs = with pkgs.python312Packages; [
      numpy
      scipy
      tqdm
      astropy
      pillow
      matplotlib
      gitpython
      hydra-core
      chuchichaestli
    ];
    src = src;
    # doCheck = false;
    meta = {
      description = "A framework for generating deep-learning SKA radio telescope & cosmological hydrodynamical simulation data.";
      meta.description.license = pkgs.lib.licenses.gpl3Plus;
    };
  };
  chuchichaestli = pkgs.python312Packages.buildPythonPackage {
    pname = "chuchichaestli";
    version = "0.2.9";
    format = "pyproject";
    build-system = with pkgs.python312Packages; [ hatchling ];
    propagatedBuildInputs = with pkgs.python312Packages; [
      numpy
      h5py
      torch
      torchmetrics
      timm
      open-clip-torch
      psutil
    ];
    doCheck = false;
    src = pkgs.fetchPypi {
      pname = "chuchichaestli";
      version = "0.2.9";
      sha256 = "sha256-mreda+JzvhHQ8RZSBgieZ301WTDQEXpyWJEDw4FQ3Dc=";
      format = "setuptools";
    };
  };
in

pythonPackage

