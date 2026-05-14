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
        # Vulkan GPU compute (Phase 7)
        vulkan-headers
        vulkan-loader
        shaderc            # provides glslc for GLSL → SPIR-V compilation
        vulkan-validation-layers
      ];

      shellHook = ''
        echo "llmtoy-zig dev shell | $(zig version)"
        # Vulkan: expose headers to @cImport and loader to the Zig linker
        export CPATH="${pkgs.vulkan-headers}/include:$CPATH"
        export LIBRARY_PATH="${pkgs.vulkan-loader}/lib:$LIBRARY_PATH"
        # Validation layers location (used at runtime when VK_INSTANCE_LAYERS is set)
        export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
      '';
    };
  };
}
