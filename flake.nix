{
  description = "llmtoy-zig: educational LLM inference in Zig";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        zig
        zls
        gdb
        perf
        hyperfine
        python3
      ];

      shellHook = ''
        echo "llmtoy-zig dev shell | $(zig version)"
      '';
    };
  };
}
