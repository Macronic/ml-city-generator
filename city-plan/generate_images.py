#!/bin/python
import tempfile
import subprocess
import sys

def get_arg(i):
	return sys.argv[i]

out = get_arg(1)
boundary_x = float(get_arg(3))
boundary_y = float(get_arg(2))
x_size = float(get_arg(5))
y_size = float(get_arg(4))
width = int(get_arg(6))
zoom = float(get_arg(7))

boundary_x2 = boundary_x + x_size
boundary_y2 = boundary_y + y_size
center_x = boundary_x + x_size/2
center_y = boundary_y + y_size/2

change_dictionary = {'<x1>' : boundary_x, '<x2>' : boundary_x2, 
					 '<y1>' : boundary_y, '<y2>' : boundary_y2, 
					 '<out>' : out, '<width>' : width, '<zoom>' : zoom, 
					 '<x>' : center_x, '<y>' : center_y}

script_template = 'template.mscript'
script_content = ''
with open(script_template) as f:
	script_content = f.read()

for (w1, w2) in change_dictionary.items():
	script_content = script_content.replace(w1, str(w2))

_, name = tempfile.mkstemp(text=True)
with open(name, 'w') as f:
	f.write(script_content)

subprocess.run(["mono", "--desktop", "Maperitive.Console.exe", name])

