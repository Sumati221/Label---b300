#!/usr/bin/env bash
# Refresh only label-guide PDFs from the organized Workspace folder.
# Usage:
#   ./refresh_label_guides_from_workspace.sh DEFAULT
set -euo pipefail

profile="${1:?Pass the Databricks CLI profile, for example DEFAULT}"
workspace_guides="/Workspace/Users/sumati.mane@philips.com/Label---b300/data/label guides"
repo_root="$(cd "$(dirname "$0")" && pwd)"
refresh_tmp="$(mktemp -d)"
trap 'rm -rf "$refresh_tmp"' EXIT

mkdir -p "$refresh_tmp/label guides"
databricks workspace export-dir "$workspace_guides" "$refresh_tmp/label guides" --overwrite --profile "$profile"
python3 "$repo_root/ingest_workspace_assets.py" --source "$refresh_tmp" --target "$repo_root"
