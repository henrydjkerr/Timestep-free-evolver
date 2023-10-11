`modules/general` contains core functionality of the program that should not vary when altering the mathematical model.

## locator.py
Read and store where individual configuration files are to be found from `settings_selection.txt`, and enable reading their contents.

###Functions:
`file_reader(str filename)`: take the name of a file, then returns a list of lists produces as follows: for each line in the file, if the line contains a colon, create a list of strings from the line's contents by splitting at the position of each colon, then append this list as a single entry in the to-be-returned list.

###Constants:
`dict location`: stores the name of each configuration file in use, keyed by a string naming their purpose.  Currently used are:
 - `parameters` stores the general numerical constants and dimensions for the model.
 - `modules` determines which modules are imported to construct and service the model in code.
 - `arrays` specifies the name, length and data type of the variable arrays needed for the model.
 - `import` optionally specifies which of these arrays should be initialised with from-file data rather than a mathematical function.  (If nothing is needed, this file still must be specified, but can be left empty.)

## Param.py
Read in the parameters config file using `locator.py`, storing the results in a dictionary.  The first entry in each line is used as the key, while the second entry is used as the value.  The data type of the value is automatically detected, with the heirarchy `int` > `float` > `str`.  Further entries on each line are ignored.

Note that `Param.py` should not be referenced directly, as `ParamPlus.py` does necessary post-processing on the dictionary.  The dictionary is then pulled forward into `Control.py` without modification, so if you already need access to `Control.py` you don't need `ParamPlus.py` as well.

###Constants:
`dict lookup`: the aforementioned dictionary of parameters.

## ParamPlus.py
Run sanity checks on the existence, data type and range of certain compulsory parameters provided by `Param.lookup`.  This may be scaled back in future to make it more generic.

Additionally, generate a small number of additional parameters derived from the existing compulsory ones.

###Constants:
`dict lookup`: the dictionary is rebound here for ease of access.

## Control.py
Read in the modules config file using `locator.py`, and import the modules requested, binding them to shorthand aliases.  Liable to be rewritten for improved generic compatibility.

Additionally, as the manager for imported modules, also contains a function for initialising arrays, since the form initialisation takes is dependent on imported modules.

###Functions:
`fill_arrays(dict arrays)`: 
