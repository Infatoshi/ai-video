# Sourced by scripts that reach the GPU host. Sets GPU (the ssh host), SONGDIR (the ACE-Step runner checkout on it)
# and REPODIR (this repo's mirror on it) from config.json (copy config.example.json and fill in your host).
_cfg="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.json"
[ -f "$_cfg" ] || { echo "config.json missing: cp config.example.json config.json and set gpu_host" >&2; exit 1; }
read -r GPU SONGDIR REPODIR < <(python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print(c['gpu_host'],c['gpu_song_dir'],c['gpu_repo_dir'])" "$_cfg")
