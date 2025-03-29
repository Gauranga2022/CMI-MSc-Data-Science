```markdown
[18 Mar 2025]  Assignment 3: Testing & Model Serving [due 1 Apr 2025]

## Unit Testing

In **score.py**, write a function with the following signature that scores a trained model on a text:

```python
def score(text: str, 
          model: sklearn.estimator, 
          threshold: float) -> (prediction: bool, propensity: float):
```

In **test.py**, write a unit test function `test_score(...)` to test the score function.
You may reload and use the best model saved during experiments in **train.ipynb** (in joblib/pkl format) for testing the score function.

You may consider the following points to construct your test cases:
- **Smoke Test:** Does the function produce some output without crashing?
- **Format Test:** Are the input/output formats/types as expected?
- **Prediction Check:** Is the prediction value 0 or 1?
- **Propensity Check:** Is the propensity score between 0 and 1?
- **Threshold Edge Cases:**
  - If you put the threshold to 0, does the prediction always become 1?
  - If you put the threshold to 1, does the prediction always become 0?
- **Content Specific Tests:**
  - On an obvious spam input text, is the prediction 1?
  - On an obvious non-spam input text, is the prediction 0?

## Flask Serving

In **app.py**, create a Flask endpoint `/score` that:
- Receives a text as a POST request.
- Returns a response in the JSON format consisting of `prediction` and `propensity`.

In **test.py**, write an integration test function `test_flask(...)` that does the following:
- Launches the Flask app using the command line (e.g., use `os.system`).
- Tests the response from the localhost endpoint.
- Closes the Flask app using the command line.

## Coverage Report

In **coverage.txt**, produce the coverage report output of the unit test and integration test using pytest.

## References

- [pytest Documentation](https://docs.pytest.org/en/8.0.x/)
- [Flask Quickstart](https://flask.palletsprojects.com/en/2.3.x/quickstart/)
```
