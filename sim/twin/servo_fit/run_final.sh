#!/usr/bin/env bash
# Final validation replays of servo_fit (one Isaac container at a time): before {} vs after (sets.json:final).
set -u
cd "$(dirname "$0")/../../.."
R="sim/twin/servo_fit/twin_replay.py"
S="--sets sim/twin/servo_fit/sets.json:final"
L=sim/twin/out/servo_fit
run() { tag=$1; shift; NAME=dlsj-twin-servo sim/twin/run.sh $R "$@" $S --tag "$tag" > $L/$tag.log 2>&1; echo "$tag $?"; }
run f_nom_delay --env grasp --play nominal --play_delay 1
run f_nom --env grasp --play nominal
run f_v1 --env grasp --play none
run f_dr --env grasp --play dr
run f_nom12 --env grasp --play nominal --n_windows 12
run f_twin12 --env twin --n_windows 12
