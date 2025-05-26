#!/bin/bash

cd ..

cd cuBB
git apply ../sera_patch/cubb.diff
cd ..

cd openairinterface5g
git apply ../sera_patch/oai.diff
cd ..
