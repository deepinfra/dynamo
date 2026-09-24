#!/usr/bin/env bash
# Build the standalone KV-cache indexer image from COMMITTED dynamo source.
#
#   ./container/build-indexer-image.sh [tag]
#
# Builds a multi-stage image (compile wheel -> slim runtime) using a git-archive
# of HEAD as the build context, so the multi-GB local target/ is never shipped
# to the docker daemon. Default tag: <REGISTRY>/dynamo-indexer:kvtest-<sha>.
# Push separately:  docker push <tag>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-localhost:30500}"
SHA="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
TAG="${1:-${REGISTRY}/dynamo-indexer:kvtest-${SHA}}"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "==> git archive ${SHA} -> build context (committed files only)"
git -C "$REPO_ROOT" archive --format=tar HEAD -o "$BUILD_DIR/dynamo-src.tar"
cp "$REPO_ROOT/container/indexer.Dockerfile" "$BUILD_DIR/Dockerfile"

echo "==> docker build $TAG"
# --ulimit nofile: BuildKit RUN steps otherwise inherit a low fd limit, which a
# big parallel cargo build exhausts ("too many open files"). The Dockerfile also
# caps CARGO_BUILD_JOBS, but raise the ceiling here too.
DOCKER_BUILDKIT=1 docker build \
  --ulimit nofile=1048576:1048576 \
  -f "$BUILD_DIR/Dockerfile" -t "$TAG" "$BUILD_DIR"

echo "==> done: $TAG"
echo "    push with: docker push $TAG"
