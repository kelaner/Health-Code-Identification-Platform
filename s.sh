python clean_temp.py
python main.py &
echo "main ok"

python begin.py &
echo "scanner system ok"


while :
do
  echo "Get"
  python scan.py
done





