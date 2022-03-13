if [ -z $1 ] ||  [ ! -f $1 ];then
    echo "Usage: %s [pos list]";
    exit;
fi

make -C .. -j4

rm -rf log;
mkdir -p log/pos log/neg


if [ -d model ];then
    let size=`ls model/*.dat|wc -l`
    if [ $size -gt 0 ];then
        mv model model`date +%H%M`
    fi
fi


mkdir model -p

./train $1 neg_list.txt new_model.dat

exit 



