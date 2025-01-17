#!/bin/bash
#

B1_NODES=100
B1_LAYERS=2
B2_NODES=200
B2_LAYERS=2
B3_NODES=400
B3_LAYERS=2
MAX_GAMES=600

# Epsilon 
MIN_EPSILON=220
MAX_EPSILON=500

# Epsilon2
EP2_VALUE=10
EP2_SCORE=1

# Epsilon step value
STEP_EPSILON=10

EPSILON=$MIN_EPSILON
while [ $EPSILON -ne $MAX_EPSILON ]; do
	python asg2.py \
		--b1_nodes $B1_NODES \
		--b1_layers $B1_LAYERS \
		--b2_nodes $B2_NODES \
		--b2_layers $B2_LAYERS \
		--b3_nodes $B3_NODES \
		--b3_layers $B3_LAYERS \
		--max_games $MAX_GAMES \
		--epsilon $EPSILON \
		--ep2_score $EP2_SCORE \
		--ep2_value $EP2_VALUE 
	EPSILON=$((EPSILON+STEP_EPSILON))
done

