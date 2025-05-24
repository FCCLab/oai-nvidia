#!/bin/bash

cd ..

cd cuBB
patch -p1 < ../sera_patch/cubb-patch.diff
cd ..

cd openairinterface5g
patch -p1 < ../sera_patch/openairinterface5g-patch.diff
cd ..
