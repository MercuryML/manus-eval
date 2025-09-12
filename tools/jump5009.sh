#!/bin/bash

ssh -v -N -D 127.0.0.1:5009 mgjump
# ssh -vNL 127.0.0.1:6888:10.0.142.148:6888 mgjump
# autossh -M 0 -f -v -N -L 6888:10.0.142.148:6888 mgjump
# test
# curl -v http://localhost:6888