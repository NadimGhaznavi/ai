#!/bin/bash
#

B1_NODES=100
B1_LAYERS=5
MAX_GAMES=1000
MAX_SCORE=50
MAX_SCORE_NUM=5

# Epsilon values
MIN_E=110
MAX_E=400

# Epsilon step value
STEP_E=10

CUR_E=$MIN_E
while [ $CUR_E != $MAX_E ]; do
	python asg.py \
		--b1_nodes $B1_NODES \
		--b1_layers $B1_LAYERS \
		--max_games $MAX_GAMES \
		--max_score $MAX_SCORE \
		--max_score_num $MAX_SCORE_NUM \
		--epsilon $CUR_E
	CUR_E=$((CUR_E+$STEP_E))
done

