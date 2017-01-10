#!/bin/bash

find -name *.ipynb -type f -print0|xargs -0 -I{} nbstripout {} 
