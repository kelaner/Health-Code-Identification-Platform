python main.py &
echo "main ok"
python scanner.py &
  echo "scanner ok"
while :
do
  sleep 3
  python scanner.py &
  echo "scanning"
done





