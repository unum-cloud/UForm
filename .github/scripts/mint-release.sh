#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 3.1.5          # stable
#     .github/scripts/mint-release.sh 3.2.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
version=$1
branch=$(git branch --show-current)
previous=$(git describe --tags --abbrev=0)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

export version
echo "$version" > VERSION
perl -pi -e 's/^(  "version": )"[^"]+",$/$1"$ENV{version}",/' package.json
perl -0pi -e 's/^( +)("name": "uform",\n\1"version": )"[^"]+",$/$1$2"$ENV{version}",/mg' package-lock.json
perl -pi -e 's/^version: .+$/version: $ENV{version}/' CITATION.cff
perl -pi -e 's/^version = "[^"]+"$/version = "$ENV{version}"/' pyproject.toml
git commit -am "Release: v$version"
.github/scripts/check-release.sh "v$version" # no hooks here, so this is what catches a missed site
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" "${flags[@]}"
