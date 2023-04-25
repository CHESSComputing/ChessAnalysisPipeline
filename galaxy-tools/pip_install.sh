#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "$0 requires the package as its only command line argument"
  exit
fi

VERSION=`python --version | awk '{split($2,a,"."); printf "%s.%s", a[1], a[2]}'`
PIPLOC=`which python | sed 's/bin\/python//g'`
PIPLOC=$PIPLOC/lib/python$VERSION/site-packages

echo "Try installing $1 to $PIPLOC"
pip install --target $PIPLOC $1 --upgrade
