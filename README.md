# DSI-24 Tools
General repository for instructions and scripts related to the [DSI-24](https://wearablesensing.com/dsi-24/) system and streaming through LSL.


Installation
---
### Install LSL
- Download the latest version at https://github.com/sccn/liblsl/releases
- Run `sudo dpkg -i {filename}.deb` in your Download folder

### Install dependencies and compile the DSI API
```bash
git clone https://github.com/thecocolab/dsi24-tools.git
cd dsi24-tools

./setup.sh
pip install -e .
```
- Clone this repository and change directory into it
- Execute the setup bash script, which downloads all dependencies and compiles the DSI API
- After running the setup script, the DSI Python API can be installed via pip


Check electrode impedance
---
- Run `sudo python scripts/impedance.py`

#### Potential problems
- If you are not not using the default COM port (`/dev/rfcomm0`) you can specify the target port through the command line as `sudo python scripts/impedance.py /path/to/port`
- If you encounter any errors, make sure `sudo python` refers to the correct python executable. You can check by comparing the outputs of `which python` and `sudo which python`, which should be the same. If `sudo python` selects the wrong executable, you can specify the full path to the correct executable you obtained from `which python` and run `sudo /path/to/python ...`


Connect the headset and start streaming
---
### Bind the device
- Turn on the headset (two short presses on the "Power button")
- If not already installed, get these packages : `sudo apt-get install bluez bluez-tools`
- Obtain the device adress : `hcitool scan`
- Bind the device : `rfcomm bind /dev/rfcomm0 xx:xx:xx:xx:xx:xx 1`

### Start LSL stream
- Run `./dsi2lsl.sh --port=/dev/rfcomm0 --lsl-stream-name=DSI_stream`

### Done
- You are now all set to receive data from the headset through an LSL stream
- To have a look the raw signal and power spectrum run `python scripts/view-signal.py`
