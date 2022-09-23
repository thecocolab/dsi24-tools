#!/bin/bash

# create a build directory
mkdir dsi24-lib
cd dsi24-lib

# download LSL App-WearableSensing client (dsi2lsl)
wget https://raw.githubusercontent.com/labstreaminglayer/App-WearableSensing/master/CLI/dsi2lsl.c
wget https://raw.githubusercontent.com/labstreaminglayer/App-WearableSensing/master/CLI/lsl_c.h

# download DSI API from the WearableSensing website
wget -O DSI-API.zip https://wearablesensing.com/wp-content/uploads/2022/09/DSI_API_v1.18.2_11172021.zip
unzip DSI-API.zip
rm DSI-API.zip

# move relevant DSI API files into the dsi2lsl CLI directory
mv DSI_API_v1.18.2_11172021/DSI.h DSI_API_v1.18.2_11172021/DSI_API_Loader.c DSI_API_v1.18.2_11172021/libDSI-Linux-x86_64.so DSI_API_v1.18.2_11172021/DSI.py .
rm -r DSI_API_v1.18.2_11172021

# compile dsi2lsl
gcc -DDSI_PLATFORM=-Linux-x86_64 -o "dsi2lsl" dsi2lsl.c DSI_API_Loader.c -ldl -llsl

# return to original directors
cd ..
