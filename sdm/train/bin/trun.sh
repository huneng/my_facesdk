if [ -z $1 ] ||  [ ! -f $1 ];then
    echo "Usage: $0 [pos list]";
    exit;
fi

make -C .. -j4

rm -r -f log    


if [ -d model ];then
    let size=`ls model/*.dat|wc -l`
    if [ $size -gt 0 ];then
        mv model model`date +%H%M`
    fi
fi

mkdir model -p

./train 0 $1 track_model.dat

exit 



