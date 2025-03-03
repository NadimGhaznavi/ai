#!/bin/bash
#

EPSILON=101
MAX_EPSILON=121
STEP=2
MAX_EPOCHS=600
MODEL=cnnr

while [ $EPSILON -lt $MAX_EPSILON ]; do
	CMD="python
AISim.py \
-ep $EPSILON \
-mo $MODEL \
-ma $MAX_EPOCHS"
	echo "Running command ($CMD)"
	$CMD
	EPSILON=$((EPSILON + STEP))
done
