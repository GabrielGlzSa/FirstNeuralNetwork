# FirstNeuralNetwork

## Description

This project leverages Docker Compose to run a Jupyter server, providing an interactive notebook for practicing Python programming skills, specifically focusing on implementing a simple neural network. An integrated service tracks changes in the notebook to enable autograding, delivering feedback whenver the notebook is saved and streamlining the learning process.

## Installation and Usage

Follow these steps to set up and run your project:

1. Clone the repository:
```
git clone https://github.com/GabrielGlzSa/FirstNeuralNetwork.git
```
2. Navigate to the project directory:
```
cd FirstNeuralNetwork
```
3. Build the Docker containers:
```
sudo docker-compose build
```
4. Make sure all running containers are stopped:
```
sudo docker-compose down
```
5. Start the Docker containers:
```
sudo docker-compose up
```

This will set up and run the Jupyter server with the notebook for practicing Python programming skills in neural networks. The autograding service will be automatically initialized to track changes and provide feedback when the notebook is saved. **Notebook must be refreshed to see feedback at the bottom.** When docker containers are started, a fresh copy of the notebook is created. Previous progress of functions is displayed in the exercise.py file. However, this file is updated whenever the notebook is saved. 

## Usage

Start the Docker containers if they are not running:
```
sudo docker-compose up
```

Access the jupyter server using the link from the terminal. Example:
```
jupyter_server | [I 2025-03-10 20:24:02.673 ServerApp]     http://127.0.0.1:8888/tree?token=928f8230ee59795f576c42b86d91386670d8eedf9f06563f
```

## Example of feedback

![Example of the feedback provided](./feedback_example.png)

## Vulnerabilities

The unittest imports the functions of the user and all the imports the user used. This can be exploited by adding code in the function for code injection. It is recommended that the code be updated to consider this by only importing libraries that are whitelisted. Consider updating the following lines in monitor.py to whitelist libraries:

```
matches = re.findall(import_pattern, script, re.MULTILINE)

# Combine the import statements.
import_matches = '\n'.join([match[0] for match in matches])
logger.info(f'Import statements:\n{import_matches}')
```


## Feedback improvement

Possible feedback is currently hardcoded into the generate_feedback function. Nonetheless, the feedback could be saved in a dataframe so that the feedback could be retrieved from the dataframe. Furthermore, the function could be replaced with a new one that uses a LLM to generate the feedback using the STDOUT of the unittest.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Provide information on how people can contact you for support, questions, or contributions. For example:

Email: gabriel.glzsa@gmail.com
GitHub: GabrielGlzSa
