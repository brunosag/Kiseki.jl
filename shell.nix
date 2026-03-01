{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "kiseki";

  buildInputs = with pkgs; [
    julia-bin
  ];

  shellHook = ''
    export JULIA_PROJECT=$PWD
    export JULIA_NUM_THREADS=auto
  '';
}
