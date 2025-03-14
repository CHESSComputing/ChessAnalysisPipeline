#!/bin/bash
##H reinstall CHAP into local venv area
##H Usage: reinstall.sh
##H
# Check if user is passing least required arguments.
if  [ "$1" == "-h" ] || [ "$1" == "-help" ] || [ "$1" == "--help" ]; then
    cat $0 | grep "^##H" | sed -e "s,##H,,g"
    exit 1
fi

echo "# uninstall CHAP from pip..."
pip uninstall --yes ChessAnalysisPipeline
echo
echo "# cleanup exists packages in dist..."
rm dist/*
echo
echo "# build new package distribution..."
python setup.py clean sdist bdist_wheel
echo
echo "# install CHAP from dist..."
pip install --no-index --find-links=dist/ ChessAnalysisPipeline
echo
echo "# check if CHAP exists in pip..."
pip list | grep Ches
