hiddenstate=(1 2 3 4)
learnrate=(5e-4 2e-4 1e-4 1e-3)
batchsize=(32 16 8)
epochs=(20 10 30 40 50 60)
for oneh in ${hiddenstate[@]}
do
  for onelr in ${learnrate[@]}
  do
    for onebs in ${batchsize[@]}
    do
        for oneep in ${epochs[@]}
        do
            echo "------------------------------"
            python LinearProbing.py \
            --hidden_state_type $oneh \
            --lr $onelr \
            --batch_size $onebs \
            --train_epoch $oneep
            echo "++++++++++++++++++++++++++++++"
        done
    done
  done
done

