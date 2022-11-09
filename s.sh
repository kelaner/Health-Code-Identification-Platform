python main.py &
echo "main ok"

python begin.py &
echo "scanner system ok"

#python write.py &
#echo "write system ok"
python clean_temp.py
while :
do
  echo "Get"
  python scanner.py
  python clean_temp.py
done





