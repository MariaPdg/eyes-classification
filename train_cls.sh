#!/bin/sh

# chmod +x script.sh
# ./script.sh &
#
echo 'Experiment 1'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 80 -vs 20
sleep 5
echo 'Experiment 2'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 70 -vs 30
sleep 5
echo 'Experiment 3'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 60 -vs 40
echo 'Experiment 4'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 50 -vs 10
echo 'Experiment 5'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 40 -vs 10
echo 'Experiment 6'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 40 -vs 20
echo 'Experiment 7'
python3 train_classifier.py  -r /home/maria/Study/VisionLabs -ts 40 -vs 30
echo 'Finish training of vae!'