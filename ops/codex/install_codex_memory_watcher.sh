#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
service_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
service_path="$service_dir/coordexp-codex-memory-watcher.service"

mkdir -p "$service_dir"

cat > "$service_path" <<EOF
[Unit]
Description=CoordExp Codex memory auto-commit watcher
Documentation=file://$repo_root/ops/codex/watch_codex_memories.sh

[Service]
Type=simple
WorkingDirectory=$repo_root
ExecStart=/usr/bin/env bash $repo_root/ops/codex/watch_codex_memories.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now coordexp-codex-memory-watcher.service

echo "installed and started: coordexp-codex-memory-watcher.service"
echo "status: systemctl --user status coordexp-codex-memory-watcher.service"
