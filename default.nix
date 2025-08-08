{
  pkgs ? import <nixpkgs> {},
  src ? ./.,
  # subdir ? "",
}:
let
  remote-chuchichaestli = builtins.fetchurl {
    url = "https://raw.githubusercontent.com/CAIIVS/chuchichaestli/refs/heads/main/default.nix";
    sha256 = "sha256:0a838l8h2qv4c95zi68r1nr8ndmn8929f53js04g3h15ii3zbskb";
  };
  chuchichaestli = pkgs.callPackage remote-chuchichaestli {
    src = pkgs.fetchFromGitHub {
      owner = "CAIIVS";
      repo = "chuchichaestli";
      rev = "main";
      sha256 = "sha256:0l5q6j7kav2lsy1pl1izqa8f31q32r7fz47qhim45gjawp838vrw";
    };
  };
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
      license = pkgs.lib.licenses.gpl3Plus;
    };
  };
in

pythonPackage

