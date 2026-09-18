#!/usr/bin/env bash
# Run the NEER Next.js frontend locally.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
npm install
npm run dev
