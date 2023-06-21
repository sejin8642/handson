#!/usr/bin/bash

text=`head -1 housing.csv`
tail -n +2 housing.csv | shuf > randomized.csv
split -d -l 1000 --additional-suffix=.csv randomized.csv data

for filename in data*.csv
do
	sed -i "1s/^/$text\n/" "${filename}"
done

rm data20.csv randomized.csv

