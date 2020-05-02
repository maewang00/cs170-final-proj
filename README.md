# CS 170 Project Spring 2020

Take a look at the project spec before you get started!

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions
  
--------------------------------------------------------------------------------------------------
PARSING INSTRUCTIONS:

1. Simply run the file directly: "python solver.py" or "python3 solver.py". This will directly call the function makeAllOutputFiles(), which will solve and build all output files into the "outputs" folder, given that there is a folder called "inputs" in the same path directory as solver.py. Make sure utils.py and parse.py are also in the same path as well.

--------------------------------------------------------------------------------------------------
Files in the same directory level for correct building:

1. solver.py
2. utils.py
3. parse.py
4. inputs folder with .in files
5. outputs folder (after running solver.py, .out files will be here)


