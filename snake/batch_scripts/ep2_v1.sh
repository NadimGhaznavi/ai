#!/bin/bash
#

B1_NODES=100
B1_LAYERS=2
B2_NODES=100
B2_LAYERS=2
B3_NODES=200
B3_LAYERS=2
MAX_GAMES=400

# Epsilon values
MIN_EP=220
MIN_EP=500


# Epsilon step value
STEP_EP=10

CUR_EP=$MIN_EP

while [ $CUR_EP -ne $MAX_EP ]; do
	python asg2.py \
		--b1_nodes $B1_NODES \
		--b1_layers $B1_LAYERS \
		--b2_nodes $B2_NODES \
		--b2_layers $B2_LAYERS \
		--b3_nodes $B3_NODES \
		--b3_layers $B3_LAYERS \
		--max_games $MAX_GAMES \
		--epsilon $CUR_EP
	CUR_EP=$((CUR_EP+STEP_EP))
done

