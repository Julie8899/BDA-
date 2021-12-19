#!/bin/sh

echo "Wait 10 seconds for Database to start up..."
sleep 10

echo "Checking baseball database existance and load data if not exists..."
mysql -h db --port 3306 --user=root --password=example -e "USE baseball" 2> /dev/null
if [ $? -eq 1 ]
then
    echo "Create and load data to database baseball ..."
    mysql -h db --port 3306 --user=root --password=example -e "CREATE DATABASE baseball"
    mysql -h db --port 3306 --user=root --password=example baseball < baseball.sql
else
    echo "baseball database already exists."
fi

echo "Creating temp_for_rolling table..."
mysql -h db --port 3306 --user=root --password=example baseball < src/temp_for_rolling.sql
echo "Creating temp_rolled_100 table..."
mysql -h db --port 3306 --user=root --password=example baseball < src/temp_rolled_100.sql
echo "Creating temp_rolled_20 table..."
mysql -h db --port 3306 --user=root --password=example baseball < src/temp_rolled_20.sql
echo "Creating temp_for_training table..."
mysql -h db --port 3306 --user=root --password=example baseball < src/temp_for_training.sql
echo "Create temp_for_training table created, ready to run FinalProject.py"

python src/FinalProject.py