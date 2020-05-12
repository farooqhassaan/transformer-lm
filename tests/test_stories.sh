#!/usr/bin/env bash
# Test preparation and training on a very small corpus
# To be run from repo root

set -ev

sp-train \
    tests/stories/ \
    tests/stories/sp-text.txt \
    tests/stories/sp-model \
    --vocab-size 5000

sp-encode \
    tests/stories/ \
    tests/stories/sp-model.model \
    tests/stories-encoded

gpt-2 \
    tests/stories-test-run/ \
    tests/stories-encoded/ \
    tests/stories/sp-model.model \
    --batch-size 8 \
    --g-accum-gradients 2 \
    --n-ctx 48 \
    --n-embed 64 \
    --n-hidden 32 \
    --n-head 4 \
    --n-layer 3 \
    --epochs 2 \
    --log-every 2 \
    --save-every 50 \
    --validate-every 100 \
    --clean

# resume training
gpt-2 \
    tests/stories-test-run/ \
    tests/stories-encoded/ \
    tests/stories/sp-model.model \
    --batch-size 8 \
    --g-accum-gradients 2 \
    --n-ctx 48 \
    --n-embed 64 \
    --n-hidden 32 \
    --n-head 4 \
    --n-layer 3 \
    --epochs 2
