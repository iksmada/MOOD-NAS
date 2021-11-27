#!/bin/bash
pwd
ssh gameleira "tar cf - /home/radamski/Documents/PC-DARTS/logs/eval*/log.txt"|tar xf -
ssh gameleira "tar cf - /home/radamski/Documents/PC-DARTS/logs/multi*/log.txt"|tar xf -
files=$(find home/ -name "log.txt")
for file in $files
do
  echo "$file"
  parent_dir=$( basename "$(dirname "$file")")
  echo "$parent_dir"
  dst="log_${parent_dir}.txt"
  echo "$dst"
  cp "$file" "$dst"
done
rm -r home/
python3 /home/raphael/master/PC-DARTS/analyse_logs.py -t log_eval-l2* -s log_multi* -o l2_loss_table.csv
python3 /home/raphael/master/PC-DARTS/analyse_logs.py -t log_eval-l1*l2*  -s log_multi* -o l1_loss_l2_fix_table.csv
l1only=($(find . -name 'log_eval-l1*' |grep -v l2))
echo "${l1only[*]}"
python3 ../analyse_logs.py -t "${l1only[@]}"  -s log_multi* -o l1_loss_table.csv


