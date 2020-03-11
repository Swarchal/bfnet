#!/bin/sh
sudo docker run \
    -v /home/swl_sur/allen_fnet_test_data:/root/projects/data \
    bayesiallen \
        --train_csv /root/projects/data/sample_data.csv \
        --image_dir /root/projects/data \
        --batch_size 2 \
        --output_path /root/projects/data \
