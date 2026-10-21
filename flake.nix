
{
  description = "Aniston - ESPN fantasy football analysis";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;

      module = { config, ... }: {
        options.services.aniston = {
          enable = lib.mkEnableOption "the Aniston Streamlit dashboard";

          port = lib.mkOption {
            type = lib.types.port;
            default = 1313;
            description = "Port for the Streamlit dashboard.";
          };

          address = lib.mkOption {
            type = lib.types.str;
            default = "127.0.0.1";
            description = "Address on which Streamlit listens.";
          };

          env = lib.mkOption {
            type = lib.types.nullOr lib.types.path;
            default = null;
            description = "Environment file that has ESPN_S2, SWID, LEAGUE_ID, and SEASON.";
          };
        };

        config = lib.mkIf config.services.aniston.enable {
          users.groups.aniston = {};

          users.users.aniston = {
            isSystemUser = true;
            group = "aniston";
            home = "/var/lib/aniston";
            createHome = false;
          };

          systemd.services.aniston = {
            description = "Aniston Streamlit dashboard";
            wantedBy = [ "multi-user.target" ];
            wants = [ "network-online.target" ];
            after = [ "network-online.target" ];

            environment = {
              HOME = "/var/lib/aniston";
              UV_PROJECT_ENVIRONMENT = "/var/lib/aniston/venv";
              UV_CACHE_DIR = "/var/cache/aniston";
              UV_PYTHON = "${pkgs.python312}/bin/python3.12";
              UV_PYTHON_DOWNLOADS = "never";
              DB_PATH = "/var/lib/aniston/aniston.duckdb";

              STREAMLIT_SERVER_ADDRESS = config.services.aniston.address;
              STREAMLIT_SERVER_PORT = toString config.services.aniston.port;
              STREAMLIT_SERVER_HEADLESS = "true";
              STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false";
            };

            serviceConfig = {
              Type = "exec";
              User = "aniston";
              Group = "aniston";
              EnvironmentFile = lib.mkIf (config.services.aniston.env != null) config.services.aniston.env;

              StateDirectory = "aniston";
              CacheDirectory = "aniston";
              WorkingDirectory = "${self}";

              ExecStart = ''
                ${pkgs.nix}/bin/nix develop ${self}#default -c \
                  ${pkgs.uv}/bin/uv run --project ${self} \
                  streamlit run ${self}/app/Home.py
              '';

              Restart = "on-failure";
              RestartSec = "10s";

              NoNewPrivileges = true;
              PrivateTmp = true;
              ProtectSystem = "strict";
              ProtectHome = true;
              ReadWritePaths = [
                "/var/lib/aniston"
                "/var/cache/aniston"
              ];
              RestrictSUIDSGID = true;
              LockPersonality = true;
              UMask = "0077";
            };
          };
        };
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python312 pkgs.uv ];

        # Native libraries needed by prebuilt Python wheels.
        LD_LIBRARY_PATH = lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];

        UV_PYTHON = "${pkgs.python312}/bin/python3.12";
        UV_PYTHON_DOWNLOADS = "never";
      };

      nixosModules.default = module;
    };
}

