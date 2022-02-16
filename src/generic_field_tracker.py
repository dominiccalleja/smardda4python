#! /bin/bash --norc
# quick script to execute the field line tracer for outboard.
import sys

run_id = sys.argv[1]


element_numbers = sys.argv[2]
if isinstance(element_numbers, str) and '.txt' in element_numbers:
    print('assuming {} is a file')
    with open(element_numbers) as f:
        element_numbers = [int(x) for x in f.read().split(',')]
else:
    element_numbers = [int(x) for x in element_numbers]

el_list_string = ''
for i in range(len(element_numbers)):
    el_list_string += '{}, '.format(element_numbers[i])


decay_length = sys.argv[3]

if len(sys.argv)>4:
    diffusion_length = sys.argv[4]
else:
    diffusion_length =0.004

n_elements = len(element_numbers)


ctl = '/Users/dominiccalleja/smardda_workflow/exec/CTL/generic_field_tracker.ctl'
powcal = '/Users/dominiccalleja/smiter/exec/powcal'

import os
with open(ctl, "r+") as f:
    old = f.read()
    txt = old.replace('<run_id>', run_id)
    txt = txt.replace('<n_elements>', str(n_elements))
    txt = txt.replace('<element_numbers>', el_list_string)

    txt = txt.replace('<decay_length>', str(decay_length))
    txt = txt.replace('<diffusion_length>', str(diffusion_length))
    output = run_id+'POW_field_track.ctl'
    fout = open(output, "wt")
    fout.write(txt)
    fout.close()
os.system('{} {}'.format(powcal, output))
