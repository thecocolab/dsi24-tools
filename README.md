# DSI-24 Tools
General repository for instructions and scripts related to the DSI-24 system.

## How to setup an LSL stream on Linux

### Install LSL and the DSI-LSL client
#### Install LSL
- Download the latest version at https://github.com/sccn/liblsl/releases
- Run `sudo dpkg -i {filename}.deb` in your Download folder
#### Install all dependencies and compile the DSI API
- Simply run the `setup.sh` script, which downloads all dependencies and compiles the API

### Connect the headset and run a stream
#### Bind the device
- Turn on the headset (two short presses on the "Power button")
- If not already installed, get these packages : `sudo apt-get install bluez bluez-tools`
- Obtain the device adress : `hcitool scan`
- Bind the device : `rfcomm bind /dev/rfcomm0 xx:xx:xx:xx:xx:xx 1`
#### Start LSL stream
- Run `./dsi2lsl.sh --port=/dev/rfcomm0 --lsl-stream-name=DSI_stream`

### Check electrode impedances
- Run `sudo python scripts/impedance.py`
  - If you are not not using the default COM port `/dev/rfcomm0` you can specify the target port through the command line as `sudo python scripts/impedance.py /path/to/port`
