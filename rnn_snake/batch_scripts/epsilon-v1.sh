#!/bin/bash
#

EPSILON=100
MAX_EPSILON=200
STEP=4
MAX_EPOCHS=300

while [ $EPSILON -lt $MAX_EPSILON ]; do
	CMD="python AISim.py -ep $EPSILON -ma $MAX_EPOCHS"
	echo "Running command ($CMD)"
	$CMD
	EPSILON=$((EPSILON + STEP))
done
