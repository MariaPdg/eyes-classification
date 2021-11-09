#!/bin/sh

# chmod +x script.sh
# ./script.sh &
#
echo 'Experiment 1'
python3 train_classifier.py  -ts 90 -vs 10
echo 'Experiment 3'
python3 train_classifier.py  -ts 80 -vs 20
echo 'Experiment 4'
python3 train_classifier.py   -ts 70 -vs 30
echo 'Experiment 5'
python3 train_classifier.py   -ts 60 -vs 40
echo 'Experiment 6'
python3 train_classifier.py   -ts 50 -vs 10
echo 'Experiment 7'
python3 train_classifier.py   -ts 40 -vs 10
echo 'Experiment 8'
python3 train_classifier.py   -ts 40 -vs 20
echo 'Experiment 9'
python3 train_classifier.py   -ts 40 -vs 30
echo 'Experiment 10'
python3 train_classifier.py   -ts 20 -vs 20
echo 'Experiment 11'
python3 train_classifier.py   -ts 30 -vs 30
echo 'Experiment 12'
python3 train_classifier.py   -ts 40 -vs 40
echo 'Experiment 13'
python3 train_classifier.py   -ts 50 -vs 50
echo 'Finish training of classifier!'