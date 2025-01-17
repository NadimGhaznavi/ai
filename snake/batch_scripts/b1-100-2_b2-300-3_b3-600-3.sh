#!/bin/bash
#

B1_NODES=100
B1_LAYERS=2
B2_NODES=300
B2_LAYERS=3
B3_NODES=600
B3_LAYERS=3
MAX_GAMES=500
MAX_SCORE=45
MAX_SCORE_NUM=5

# Epsilon values
MIN_E=250
MAX_E=500

# Epsilon step value
STEP_E=10

CUR_E=$MIN_E
while [ $CUR_E != $MAX_E ]; do
	python asg.py \
		--b1_nodes $B1_NODES \
		--b1_layers $B1_LAYERS \
		--b2_nodes $B2_NODES \
		--b2_layers $B2_LAYERS \
		--b3_nodes $B3_NODES \
		--b3_layers $B3_LAYERS \
		--max_games $MAX_GAMES \
		--max_score $MAX_SCORE \
		--max_score_num $MAX_SCORE_NUM \
		--epsilon $CUR_E
	CUR_E=$((CUR_E+$STEP_E))
done

