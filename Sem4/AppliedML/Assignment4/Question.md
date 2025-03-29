[18 Mar 2025]  Assignment 4: Containerization & Continuous Integration [due 8 Apr 2025]

## Containerization

- **Docker Container for Flask App:**
  - Create a Docker container for the Flask app created in Assignment 3.
  - Create a `Dockerfile` which contains the instructions to build the container, including:
    - Installing the dependencies.
    - Copying `app.py` and `score.py`.
    - Launching the app by running `python app.py` upon entry.
- **Building and Running:**
  - Build the Docker image using the `Dockerfile`.
  - Run the Docker container with appropriate port bindings.
- **Docker Testing:**
  - In `test.py`, write a function `test_docker(...)` which does the following:
    - Launches the Docker container using the command line (e.g., using `os.system(...)` along with `docker build` and `docker run` commands).
    - Sends a request to the localhost endpoint `/score` (using the `requests` library) for a sample text.
    - Checks if the response is as expected.
    - Closes the Docker container.
- **Coverage:**
  - In `coverage.txt`, produce the coverage report using pytest for the tests in `test.py`.

## Continuous Integration

- **Pre-commit Git Hook:**
  - Write a pre-commit git hook that will run `test.py` automatically every time you try to commit the code to your local `main` branch.
  - Copy and push this pre-commit git hook file to your git repository.

## References

- [Docker Curriculum](https://docker-curriculum.com/)
- [Tutorialspoint Docker Overview](https://www.tutorialspoint.com/docker/docker_overview.htm)
- [How to Dockerize a Flask App](https://www.freecodecamp.org/news/how-to-dockerize-a-flask-app/)
- [GitHooks](https://githooks.com/)
- [A Simple Git Hook for Your Python Projects](https://www.giacomodebidda.com/posts/a-simple-git-hook-for-your-python-projects/)

