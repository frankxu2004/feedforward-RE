#!/bin/bash

# Tunnel through an NLP access machine to a machine running
# Tensorboard and open Tensorboard in a browser.
#
# Author: Will Monroe

INNER_HOST=jagupard9
INNER_PORT=8866
OUTER_HOST=zyh@jacob.stanford.edu
OUTER_PORT=8866

echo "Tunnelling to $INNER_HOST:$INNER_PORT via $OUTER_HOST:$OUTER_PORT ..."
ssh -M -S tboard -fnN -L "$OUTER_PORT:$INNER_HOST:$INNER_PORT" "$OUTER_HOST" "$@"
ssh -S tboard -O check "$OUTER_HOST"

#chromium-browser http://localhost:${OUTER_PORT}/ &disown

echo "Press Ctrl-C to kill the tunnel."
trap : SIGINT
cat > /dev/null
trap - SIGINT

echo "Killing..."
ssh -S tboard -O exit "$OUTER_HOST"
