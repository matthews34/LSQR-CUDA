if [ -z "$STARTING_SIZE" ]; then
    STARTING_SIZE="100"
fi

if [ -z "$MAX_SIZE" ]; then
    MAX_SIZE="1000"
fi

if [ -z "$SIZE_STEP" ]; then
    SIZE_STEP="100"
fi

if [[ ! -f "LSQR_CPU" || ! -f "LSQR_GPU" ]]; then
    ./compile_all.sh
fi

if [[ ! -d "data" || -z "$(ls -A data)" ]]; then
    ./gen_data.sh
fi

mkdir -p results

for i in $(seq $STARTING_SIZE $SIZE_STEP $MAX_SIZE)
do
    for j in $(seq $STARTING_SIZE $SIZE_STEP $MAX_SIZE)
    do
        echo "matrix_${i}_${j} vector_$i"
        ./LSQR_CPU data/matrix_${i}_${j} data/vector_$i > results/matrix_${i}_${j}_cpu
        ./LSQR_GPU data/matrix_${i}_${j} data/vector_$i > results/matrix_${i}_${j}_gpu
    done
done
