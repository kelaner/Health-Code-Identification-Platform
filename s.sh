python main.py &
echo "main ok"

python begin.py &
echo "scanner system ok"

python clean_temp.py
while :
do
  echo "Get"
  python scanner.py
done





