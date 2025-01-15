#!/bin/bash
#
#b1n_100-b1l_4-m_600_e50-to-250.sh

B1_NODES=100
B1_LAYERS=4
MAX_GAMES=600

# Epsilon values
MIN_E=50
MAX_E=250

# Epsilon step value
STEP_E=10

CUR_E=$MIN_E
while [ $CUR_E != $MAX_E ]; do
	python asg.py --b1_nodes 100 --b1_layers 4 --max_games 600 --epsilon $CUR_E
	CUR_E=$((CUR_E+$STEP_E))
done

