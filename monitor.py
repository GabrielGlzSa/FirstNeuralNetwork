import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import subprocess

import nbformat
from nbconvert import PythonExporter

import re

import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("monitor.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

ROOT_DIRECTORY = os.getcwd()
notebook_name = 'exercise'

logger.info(f'Current working directory: {ROOT_DIRECTORY} to watch for changes in {notebook_name}')
DIRECTORY_TO_WATCH = os.path.join(ROOT_DIRECTORY, 'notebooks/')

# Placeholder for order of the functions for feedback.
func_order = None

# Paths
notebook_path = os.path.join(DIRECTORY_TO_WATCH, notebook_name+'.ipynb')
converted_nb_path = os.path.join(DIRECTORY_TO_WATCH, notebook_name+'.py')

def convert_notebook_to_script(notebook_path, converted_nb_path):
    logger.info(f'Reading notebook from path: {notebook_path}')
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
            notebook = nbformat.reads(content, as_version=4)
    except nbformat.reader.NotJSONError as e:
        logger.error(f"Could not load jupyter notebook '{notebook_path}' using JSON due to: {e}")
        return False
    except Exception as e:
        logger.error(f'An error occurred: {e}')
        return False
    
    
    # Convert notebook to Python script
    python_exporter = PythonExporter()
    script, _ = python_exporter.from_notebook_node(notebook)
    
    # Find all import statements in the script.
    import_pattern = r'^\s*(import\s+(\w+(\s+as\s+\w+)?(,\s*\w+(\s+as\s+\w+)?)*)|from\s+\w+(\.\w+)*\s+import\s+\w+(\s+as\s+\w+)?(,\s*\w+(\s+as\s+\w+)?)*)\s*$'
    """
    CAUTION: THIS REGEX IMPORTS ALL LIBRARIES FROM THE STUDENT'S CODE TO BE ABLE TO RUN IT.
    IT CAN BE USED FOR CODE INJECTION ATTACKS. IT WOULD BE BETTER TO RESTRICT THE USAGE OF LIBRARIES TO A WHITELIST.
    """
    
    matches = re.findall(import_pattern, script, re.MULTILINE)

    # Combine the import statements.
    import_matches = '\n'.join([match[0] for match in matches])
    logger.info(f'Import statements:\n{import_matches}')

    # Find all functions in the script.
    func_def_pattern = r"def\s+\w+\(.*?\):.*?return\s+.*?\n\n"
    func_matches = re.findall(func_def_pattern, script, re.DOTALL)

    global func_order

    func_order = []
    for func in func_matches:
        start_idx = func.index('f')
        end_def = func.index('(')
        func_order.append(func[start_idx+2:end_def])


    # Combine the import statements and function definitions.
    summarized_script = import_matches + "\n" + "\n".join(func_matches)

    # Save the summarized script to a .py file
    logger.info(f'Saving .py in path: {converted_nb_path}')
    with open(converted_nb_path, 'w') as f:
        f.write(summarized_script)

    
    return True

    
    
def edit_last_markdown_cell(notebook_path, new_content):
    """
    Edits the content of the last markdown cell in a Jupyter notebook
    to provide feedback to the student. 
    Args:
        notebook_path (str): The path to the Jupyter notebook
        new_content (str): The new content to be added to the last markdown cell
    Returns:
        None
    """
    # Load the existing notebook
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
    
    # Get the last cell
    last_cell = nb.cells[-1] if nb.cells else None

    # Check if the last cell is a Markdown cell
    if last_cell and last_cell.cell_type == 'markdown':
        # Edit the content of the last markdown cell
        last_cell.source = new_content
        logger.info("Last markdown cell updated.")
    else:
        new_cell = nbformat.v4.new_markdown_cell(new_content)
        # Append the new cell to the notebook
        nb.cells.append(new_cell)

    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)




def generate_feedback(matches, n_tests, n_failed_tests, n_error_tests):
    """
    Hardcoded feedback in markdown based on the results of the tests.
    Args:
        matches (list): List of dictionaries containing the test name and the error message.
        n_tests (int): Number of tests run.
        n_failed_tests (int): Number of tests that failed due to logic.
        n_error_tests (int): Number of tests that failed due to python errors.
    Returns:
        str: The feedback to be provided to the student.
    """

    logger.debug(matches)
    feedback = "# Results of test  \n"
    feedback += f"Number of Exercises: {n_tests}  \n\n"
    feedback += f"Number of not passed exercises: {n_failed_tests} \n\n"
    feedback += f"Number of unscored exercises due to coding errors: {n_error_tests}  \n\n"
    
    feedback += f"## **You passed {n_tests-(n_failed_tests+n_error_tests)} out of {n_tests} exercises.**  \n\n"

    if matches:
        feedback += "##  Tests you have not passed: \n"

    for match in matches:
        logger.info(f"Function {match['func_name']} produced {match['status']} due to {match['error']}: {match['msg']}")
        feedback += f"###  **{match['func_name']}** \n"

        if match['func_name'] == 'relu_function':
            if match['error'] == 'AssertionError':
                feedback += "Although function runs, the result is different than expected. \n"
                if '-' in match['msg']:
                    feedback += " - The ReLU function should return 0 for $x < 0$. \n"
                else:
                    feedback += " - The ReLU function should return $x$ for $x >= 0$. \n"


        elif match['func_name'] == 'relu_function_layer':
            if match['error'] == 'AttributeError':
                feedback += f" - Encountered error was {match['msg']}. \n"
                feedback += f" - Please verify you are returning a numpy array. \n"
            elif match['error'] == 'AssertionError':
                if 'Tuples differ' in match['msg']:
                    feedback += f" - Function is returning a different array shape than expected. \n"
                elif 'Arrays are not equal' in match['msg']:
                    feedback += f" - Function is returning correct shape. However, content is different than expected. \n"

        elif match['func_name'] == 'calculate_neuron_logit':
            if match['error'] == 'AssertionError':
                # Logit is missing bias.
                if 'np.float64(0.24000000000000002)' in match['msg']:
                    feedback += " - The Dot product is correct. You are missing one operation after the dot product. \n"
                else:
                    feedback += " - The Dot product or the bias is incorrect . Please verify your procedure. \n"
            elif match['error'] == 'ValueError':
                if 'array' in match['msg']:
                    feedback += f" - Function must return an scalar, not a list nor array. \n"
                    

        elif match['func_name'] == 'calculate_logits_layer':
            if match['error'] == 'AttributeError':
                feedback += f" - Encountered error was {match['msg']}. \n"
                feedback += f" - Please verify you are returning a numpy array. \n"
            elif match['error'] == 'AssertionError':
                if 'Tuples differ' in match['msg']:
                    feedback += f" - Function returns an array of different shape than the input. \n"
                elif 'Arrays are not equal' in match['msg']:
                    feedback += f" - Function returns an array of expected shape. However, the content differs from expected. \n"


        elif match['func_name'] == 'softmax_layer':
            if match['error'] == 'AttributeError':
                feedback += f" - Encountered error was {match['msg']}. \n"
                feedback += f" - Please verify you are returning a numpy array. \n"
            elif match['error'] == 'AssertionError':
                if "DESIRED: 1.0\n" in match['msg']:
                    feedback += f"The sum of probabilities in array is not 1.0. Make sure the sum of probabilities es 1.0. \n"
                else:
                    feedback += f"The sum of probabilities in array is 1.0. Nonetheless, the probabilities assigned to the neurons are wrong. \n"

        elif match['func_name'] == 'weight_initialization':
            if match['error'] == 'AttributeError':
                feedback += f" - Encountered error was {match['msg']}. \n"
                feedback += f" - Please verify you are returning a numpy array. \n"
            elif match['error'] == 'TypeError':
                feedback += f" - The function should return 6 numpy arrays with the randomly initialized weights. \n"
            elif match['error'] == 'AssertionError':
                if 'Tuples differ' in match['msg']:
                    feedback += f" - One or more weights were initialized with the wrong shape. \n"
                    feedback += f" - Remember the weights of a layer depends in the input shape of that layer and the number of neurons in the layer. \n"

        elif match['func_name'] == 'neural_network':
            if match['error'] == 'AttributeError':
                feedback += f" - Encountered error was {match['msg']}. \n"
                feedback += f" - Please verify you are returning the output of each layer. \n"
            elif match['error'] == 'TypeError':
                if "cannot unpack non-iterable" in match['msg']:
                    feedback += f" -  The function should return three numpy arrays containing the outputs of each layer. \n"

        # feedback += f"\n{match['error']}: {match['msg']} \n"

    return feedback


def run_tests():
    """
    Runs the unittest and retrieves the logs of the tests. Then it generates the feedback depending on the failed tests.
    Although the feedback is hardcoded, an LLM could provide a 
    more sophisticated feedback based on the tests results.
    

    Example of not passed test:
    notebook_monitor | FAIL: test_relu_function (test_answer.TestExercises.test_relu_function)
    notebook_monitor | ----------------------------------------------------------------------
    notebook_monitor | Traceback (most recent call last):
    notebook_monitor |   File "/app/test_answer.py", line 6, in test_relu_function
    notebook_monitor |     self.assertEqual(relu_function(-3), 0)
    notebook_monitor | AssertionError: -4 != 0
    notebook_monitor | 
    notebook_monitor | ----------------------------------------------------------------------

    """

    # Run unit tests by running test_answer.py
    try:
        result = subprocess.run(["python", "test_answer.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f"An error occurred while running the tests: {e}")
        return 
    
    stdout = result.stdout
    logger.info('STDOUT:\n'+stdout)

    if 'SyntaxError' not in stdout: 
    
        # Find all failed tests due to errors in the code or assertion errors.
        test_pattern = r'^(?P<status>ERROR|FAIL): test_(?P<func_name>[\w+\_]+)\s\(|^(?P<error>\w*Error):\s(?P<msg>[\w\.\(\)\s\!\-:\/\']+\={0,1}\s[\w\.\-]*)'
        test_matches = re.findall(test_pattern, stdout, re.MULTILINE)

        test_output_matches = []

        n_failed_tests = 0
        n_error_tests = 0

        global func_order

        for first_match, second_match in zip(test_matches[::2], test_matches[1::2]):

            test_output_matches.append(
                {
                'status': first_match[0], 
                'func_name': first_match[1], 
                'error': second_match[2], 
                'msg': second_match[3].replace('\n\n', '\n'),
                'order': func_order.index(first_match[1])
                }
            )
            if first_match[0]=='FAIL':
                n_failed_tests +=1
            elif first_match[0]=='ERROR':
                n_error_tests +=1


        # Sort functions by order in notebook.
        test_output_matches = sorted(test_output_matches, key= lambda x: x['order'])


        # Find number of tests run.
        n_tests_patterns = r"Ran (\d+) test"
        n_tests = int(re.search(n_tests_patterns, stdout).group(1))

        logger.info(f"Number of tests: {n_tests}")

        # Number of tests not passed due to asserts.
        logger.info(f"Number of failed tests: {n_failed_tests}")
        

        # Number of tests not passed due to python errors in function or test.
        logger.info(f"Number of python errors tests: {n_error_tests}")

        # Generate feedback in markdown format.
        cell_content = generate_feedback(test_output_matches, n_tests, n_failed_tests, n_error_tests) 
    else:
        cell_content = "# Syntax error in your functions. Please fix it before being scored."         
    
    #Update the last cell in the notebook with the feedback.
    edit_last_markdown_cell(notebook_path, cell_content)

class NotebookHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = 0
    
    def on_modified(self, event):
        current_time = time.time()
        logger.info('Time passed since last trigger is {:.2f} seconds'.format(current_time - self.last_modified))
        if event.src_path.endswith('/'+notebook_name+'.ipynb'):
            if (current_time - self.last_modified > 1):
            
            
                logger.info("Notebook saved. Scoring the solution...")
                # Here, add your code to score the notebook.
                time.sleep(1)
                success_flag = convert_notebook_to_script(notebook_path, converted_nb_path)
                if success_flag:
                    logger.info(f"Running unit tests due to modification in {event.src_path}")
                    run_tests()
                else:
                    logger.info('Could not run test.')
                
                self.last_modified = current_time
            elif current_time - self.last_modified <0:
                logger.info('Time is negative. Resetting the last modified time.')
                self.last_modified = current_time
                



if __name__ == "__main__":
    event_handler = NotebookHandler()
    observer = Observer()
    observer.schedule(event_handler, DIRECTORY_TO_WATCH, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

