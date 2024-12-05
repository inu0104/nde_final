#!/bin/bash

# Set different combinations of hyperparameters to try
for n_layers in 3 4 5
do
  for vNetWidth in 64 128 256
  do
    for learning_rate in 0.001 0.005 0.01
    do
      for batch_size in 100 200
      do
        echo "Running with n_layers=$n_layers, vNetWidth=$vNetWidth, learning_rate=$learning_rate, batch_size=$batch_size"
        python main.py \
          --n_layers $n_layers \
          --vNetWidth $vNetWidth \
          --learning_rate $learning_rate \
          --batch_size $batch_size \
          --n_epochs 50 \
          --activation relu \
          --activation_output softplus \
          --device cuda
      done
    done
  done
done