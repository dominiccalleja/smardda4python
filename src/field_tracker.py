#! /bin/bash --norc
# quick script to execute the field line tracer for outboard.
import sys

run_id = sys.argv[1]

ctl = '/Users/dominiccalleja/demo_misalighnments/exec/CTL/outboard_profile_fieldtrack.ctl'
powcal = '/Users/dominiccalleja/smiter/exec/powcal'

import os
with open(ctl, "r+") as f:
    old = f.read()
    txt = old.replace('<run_id>', run_id)

    output = run_id+'POW_field_track.ctl'
    fout = open(output, "wt")
    fout.write(txt)
    fout.close()
os.system('{} {}'.format(powcal, output))
