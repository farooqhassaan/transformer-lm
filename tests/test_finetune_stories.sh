#!/usr/bin/env bash
# Test preparation and training on a very small corpus
# To be run from repo root

set -ev

sp-encode \
    tests/stories-f/ \
    tests/stories-f/sp.model \
    tests/stories-encoded-f

#gpt-2 \
#    tests/stories-test-run/ \
#    tests/stories-encoded-f/ \
#    tests/stories/sp-model.model \
#    --batch-size 8 \
#    --g-accum-gradients 1 \
#    --n-ctx 1024 \
#    --n-embed 1024 \
#    --n-hidden 1024 \
#    --n-head 16 \
#    --n-layer 24 \
#    --epochs 50 \
#    --log-every 2 \
#    --save-every 50 \
#    --validate-every 100 \
#    --clean

# resume training
gpt-2 \
   tests/stories-test-run-f/ \
   tests/stories-encoded-f/ \
   tests/stories-f/sp-model.model \
   --batch-size 8 \
   --g-accum-gradients 1 \
   --n-ctx 1024 \
   --n-embed 1024 \
   --n-hidden 1024 \
   --n-head 16 \
   --n-layer 24 \
   --epochs 2
