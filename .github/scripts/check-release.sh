#!/usr/bin/env bash
# Checks that HEAD is the release commit for a tag and that every version site carries its version.
#
#     .github/scripts/check-release.sh v3.1.5
set -euo pipefail
tag=$1 version=${1#v}
subject=$(git log -1 --format=%s)
if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]]; then
    echo "check-release: a release tag is exactly 'vX.Y.Z' or 'vX.Y.Z-rc.N', got '$tag'"
    exit 1
fi
if [ "$subject" != "Release: $tag" ]; then
    echo "check-release: $tag must sit on 'Release: $tag', but HEAD is '$subject'"
    exit 1
fi
stale=""
[ "$(cat VERSION)" = "$version" ] || stale="$stale VERSION"
grep -qxF "  \"version\": \"$version\"," package.json || stale="$stale package.json"
# Both the lockfile root and its `packages[""]` entry name the package itself.
[ -z "$(awk -v want="\"version\": \"$version\"," '/^ +"name": "uform",$/ { getline; sub(/^ +/, ""); if ($0 != want) print }' package-lock.json)" ] ||
    stale="$stale package-lock.json"
grep -qxF "version: $version" CITATION.cff || stale="$stale CITATION.cff"
grep -qxF "version = \"$version\"" pyproject.toml || stale="$stale pyproject.toml"
if [ -n "$stale" ]; then
    echo "check-release: these files don't carry $version:$stale"
    exit 1
fi
