#!/bin/bash

if [ -d lib ]
then
	# run dsi2lsl program
	sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib ./lib/dsi2lsl $@
else
	# setup was not completed
	echo "Please run setup.sh first"
fi
