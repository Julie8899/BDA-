#!/bin/sh
DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"
QUERY_AVERAGE="command.sql"
echo "Creating database if not exists..."
mysql -h db --port 3306 --user=root --password=example -e "CREATE DATABASE IF NOT EXISTS ${DATABASE_TO_COPY_INTO};"
echo "Using baseball database...and loading baseball"
mysql -h db --port 3306 --user=root --password=example -e "USE ${DATABASE_TO_COPY_INTO};source ${DATABASE_FILE};"
# mysql -h db --port 3306 --user=root --password=example -e "USE baseball;source baseball.sql;"
echo "Run command and send results..."
mysql -h db --port 3306 --user=root --password=example -e "USE ${DATABASE_TO_COPY_INTO};source ${QUERY_AVERAGE};" > /results/average.txt