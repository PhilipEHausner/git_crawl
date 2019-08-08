import os
import re
import keyword

# Regular expressions, we need to find Python variable names
variable_start_py = re.compile((r"^[a-zA-Z_]"))
variable_all_py = re.compile((r"^[a-zA-Z_][a-zA-Z_0-9]*"))

variable_names = []
# iterate over all directories
for subdir, dirs, files in os.walk('../output/python'):
    for file in files:
        filepath = os.path.join(subdir, file)
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            content = f.readlines()
        for line in content:
            # to get rid of tabs and spaces at start and/or end of line
            line = line.strip()
            # re to find only vars that start with a valid character and have an assignment
            if re.match(variable_start_py, line) and '=' in line:
                # line 1: split at assignment and only take first part
                # line 2: remove spaces
                # line 3: detect arrays and remove trailing special characters
                # line 4: for classes, packages, self, select only most granular element
                variable = line.split('=')[0] \
                    .strip().split(' ')[0] \
                    .split('.')[-1]
                variable = re.match(variable_all_py, variable)
                if variable:
                    variable = variable[0]
                else:
                    continue
                # check that found variable is not a keyword or function call
                if variable not in keyword.kwlist:
                    variable_names.append(variable)

with open('./test_vars.txt', 'w') as f:
    f.write('\n'.join(variable_names))