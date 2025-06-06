#1/bin/bash

SCRIPT_DIR=$(dirname "$0")
echo "Running xApp monitoring script from $SCRIPT_DIR"

export PYTHONPATH=$SCRIPT_DIR/../../openairinterface5g/openair2/E2AP/flexric/build/examples/xApp/python3:$PYTHONPATH
export LD_LIBRARY_PATH=$SCRIPT_DIR/../../openairinterface5g/openair2/E2AP/flexric/build/examples/xApp/python3:$LD_LIBRARY_PATH
export PATH=$SCRIPT_DIR/../../openairinterface5g/openair2/E2AP/flexric/build/examples/xApp/python3:$PATH

python3 $SCRIPT_DIR/xapp_mac_rlc_pdcp_gtp_moni.py
