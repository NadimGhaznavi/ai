#!/bin/bash
#

EPSILON=30
MAX_EPSILON=85
STEP=1
MAX_EPOCHS=100

while [ $EPSILON -lt $MAX_EPSILON ]; do
	CMD="python AISim.py -ep $EPSILON -ma $MAX_EPOCHS -mo rnn"
	echo "Running command ($CMD)"
	$CMD
	EPSILON=$((EPSILON + STEP))
done
