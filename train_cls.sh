#!/bin/sh

# chmod +x script.sh
# ./script.sh &
#
echo 'Experiment 1: -ts 20 -vs 20'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 20 -vs 20
sleep 5
echo 'Experiment 2: -ts 30 -vs 30'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 30 -vs 30
sleep 5
echo 'Experiment 3: -ts 40 -vs 40'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 40 -vs 40
echo 'Experiment 4: -ts 50 -vs 50'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 50 -vs 50
echo 'Finish training of vae!'