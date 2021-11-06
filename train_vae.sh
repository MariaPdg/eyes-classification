#!/bin/sh

# chmod +x script.sh
# ./script.sh &

echo 'Experiment 1: beta=0.1'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.1 -bs 64
sleep 5
echo 'Experiment 2: beta=0.01'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.01 -bs 64
sleep 5
echo 'Experiment 3: beta=0.001'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.001 -bs 64
echo 'Experiment 4: beta=0.1'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.1 -bs 128
sleep 5
echo 'Experiment 5: beta=0.01'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.01 -bs 128
sleep 5
echo 'Experiment 6: beta=0.001'
python3 train_vae.py  -r /home/maria/Study/VisionLabs -be 0.001 -bs 128
echo 'Finish training of vae!'
