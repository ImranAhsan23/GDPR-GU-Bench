#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-external_repos}
mkdir -p "$ROOT"
cd "$ROOT"
clone_or_pull () {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[pull] $dir"
    git -C "$dir" pull --ff-only || true
  else
    echo "[clone] $url -> $dir"
    git clone "$url" "$dir"
  fi
}
clone_or_pull https://github.com/MinChen00/Graph-Unlearning.git GraphEraser_official
clone_or_pull https://github.com/wujcan/GIF-torch.git GIF_torch_official
clone_or_pull https://github.com/Sumsky21/Unlink-to-Unlearn.git UtU_official
clone_or_pull https://github.com/yushundong/IDEA.git IDEA_official
clone_or_pull https://github.com/Aliezzz/AGU.git AGU_official
clone_or_pull https://github.com/bwfan-bit/OpenGU.git OpenGU_official

echo "\nOfficial repositories cloned into $ROOT."
echo "Run 'git -C <repo> rev-parse HEAD' and record commit hashes in your paper/reproducibility log."
