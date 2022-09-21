# DSI-24 Tools
General repository for instructions and scripts related to the DSI-24 system.

## How to setup an LSL stream on Linux

### Install LSL and the DSI-LSL client
#### Install LSL
- Download the latest version at https://github.com/sccn/liblsl/releases
- Run ```sudo dpkg -i {filename}.deb``` in your Download folder
#### Compile client
- Download the [LSL-DSI app](https://github.com/labstreaminglayer/App-WearableSensing) : ```git clone git@github.com:labstreaminglayer/App-WearableSensing```
- Download and extract the [DSI API](https://wearablesensing.com/wp-content/uploads/2022/09/DSI_API_v1.18.2_11172021.zip)
- Move these files from the DSI API folder to the App-WearableSensing/CLI folder : ```DSI.h``` ```DSI_API_Loader.c``` ```libDSI-Linux-x86_64.so``` ```DSI.py```
- Go to the App-WearableSensing/CLI folder : ```cd App-WearableSensing/CLI```
- Compile the executable : ```gcc -DDSI_PLATFORM=-Linux-x86_64  -o "dsi2lsl" dsi2lsl.c  DSI_API_Loader.c -ldl -llsl```

### Connect the headset and run a stream
#### Bind the device
- Turn on the headset (two short presses on the "Power button")
- If not already installed, get these packages : ```sudo apt-get install bluez bluez-tools```
- Obtain the device adress : ```hcitool scan```
- Bind the device : ```rfcomm bind /dev/rfcomm0 xx:xx:xx:xx:xx:xx 1```
#### Start LSL stream
- Move to the App-WearableSensing/CLI folder : ```cd App-WearableSensing/CLI```
- Run ```sudo LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./dsi2lsl --port=/dev/rfcomm0 --lsl-stream-name=DSI_stream```

### Check electrode impedances
- Move the `impedance.py` file from this repository to `App-WearableSensing/CLI` (requires the DSI Python API to be present in the same directory)
- Run `sudo python impedance.py`
  - If you are not not using the default COM port `/dev/rfcomm0` you can specify the target port through the command line as `sudo python impedance.py /path/to/port`
