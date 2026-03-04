{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "kiseki";

  buildInputs = with pkgs; [
    julia-bin
    pprof
    graphviz
  ];

  shellHook = ''
    export JULIA_PROJECT=$PWD
    export JULIA_NUM_THREADS=auto
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
  '';
}
