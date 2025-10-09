#!/bin/bash
#

EPSILON=150
MAX_EPSILON=170
STEP=1
MAX_EPOCHS=500

while [ $EPSILON -lt $MAX_EPSILON ]; do
	CMD="python
AISim.py \
-ep $EPSILON \
-ma $MAX_EPOCHS"
	echo "Running command ($CMD)"
	$CMD
	EPSILON=$((EPSILON + STEP))
done
