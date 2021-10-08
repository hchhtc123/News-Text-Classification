python train1_robert.py
python train1_nezha.py
python train1_skep.py
python mergesim1.py

python train2_robert.py
python train2_nezha.py
python train2_skep.py
python mergesim2.py

python train3_robert.py
python train3_nezha.py
python train3_skep.py
python mergesim3.py

python train4_robert.py
python train4_nezha.py
python train4_skep.py

python read.py
python kaggle_vote.py "./_*.csv" "./merge.csv" "weighted"
python correct.py

zip 'result.zip' 'result.txt'