#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_file="$script_dir/mooncake_connector_proxy.go"
output_file=${1:-"$script_dir/mooncake_connector_proxy"}

case $(uname -m) in
	x86_64)  GOARCH="amd64" ;;
	aarch64) GOARCH="arm64" ;;
	*) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

CGO_ENABLED=0 GOOS=linux GOARCH="$GOARCH" go build \
    -trimpath \
    -ldflags="-s -w" \
    -o "$output_file" \
    "$source_file"

echo "Built $output_file for linux/$GOARCH"
