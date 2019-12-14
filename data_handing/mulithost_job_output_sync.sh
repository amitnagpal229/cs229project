#!/bin/sh

for file in `ls /home/anagpal/project/pose/videos/training/*_features.* /home/anagpal/project/pose/videos/training/*_output.mp4`
do
  filename=`echo $file | awk -F'/' '{print $NF}'`
  [ -f $filename ] || cp $file .
done

for file in `ssh tesla102.sieve.bf1.yahoo.com "ls /home/anagpal/project/pose/videos/training/*_features.* /home/anagpal/project/pose/videos/training/*_output.mp4"`
do
  filename=`echo $file | awk -F'/' '{print $NF}'`
  [ -f $filename ] || scp tesla102.sieve.bf1.yahoo.com:$file .
done

for file in `ssh tesla101.sieve.bf1.yahoo.com "ls /home/anagpal/project/pose/videos/training/*_features.* /home/anagpal/project/pose/videos/training/*_output.mp4"`
do
  filename=`echo $file | awk -F'/' '{print $NF}'`
  [ -f $filename ] || scp tesla101.sieve.bf1.yahoo.com:$file .
done
