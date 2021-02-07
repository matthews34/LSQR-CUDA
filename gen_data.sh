if [ -z "$STARTING_SIZE" ]; then
    STARTING_SIZE="100"
fi

if [ -z "$MAX_SIZE" ]; then
    MAX_SIZE="1000"
fi

if [ -z "$SIZE_STEP" ]; then
    SIZE_STEP="100"
fi

if [ -z "$SPARSITY" ]; then
    SPARSITY="0.8"
fi


if [ ! -f "create_sparse_matrix" ]; then
    g++ -o create_sparse_matrix testing/create_sparse_matrix.cpp -std=c++11
fi

if [ ! -f "create_vector" ]; then
    g++ -o create_vector testing/create_vector.cpp -std=c++11
fi

mkdir -p data
cd data

for i in $(seq $STARTING_SIZE $SIZE_STEP $MAX_SIZE)
do
    ../create_vector $i
    for j in $(seq $STARTING_SIZE $SIZE_STEP $MAX_SIZE)
    do
        ../create_sparse_matrix $i $j ${SPARSITY}
    done
done