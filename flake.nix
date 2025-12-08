{
  description = "Dev shell for codex_dspy (Python 3.13 + uv).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = { system, pkgs, ... }:
        let
          python = pkgs.python313;
          cxxLib = pkgs.stdenv.cc.cc.lib;
          ldLibPath = pkgs.lib.makeLibraryPath [
            cxxLib       # libstdc++.so.6 for tokenizers / rust-backed wheels
            pkgs.zlib
            pkgs.openssl
          ];
        in {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              python
              uv
              git
              pkg-config
              openssl
              libffi
              zlib
              cxxLib
            ];

            env =
              {
                UV_PYTHON = "${python.interpreter}";
                UV_PYTHON_DOWNLOADS = "never";
              }
              // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
                # Only needed on Linux where libstdc++.so.6 must be discoverable for tokenizers.
                LD_LIBRARY_PATH = "${ldLibPath}";
              };

            shellHook = ''
              # Create/sync uv environment for this project
              if [ -f pyproject.toml ]; then
                echo "[devshell] syncing uv env..."
                if ! uv sync --frozen; then
                  uv sync
                fi
              fi
            '';
          };
        };
    };
}
