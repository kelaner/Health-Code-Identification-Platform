python clean_temp.py
echo "clean_temp ok"
python main.py &
echo "one ok"
sleep 10
while :
do
    python change.py
    echo "change ok"
    python word.py
    echo "word ok"
    python clean_temp.py
    echo "clean_temp ok"
done



