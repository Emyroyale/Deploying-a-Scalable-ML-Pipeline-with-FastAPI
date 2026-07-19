# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier built using scikit-learn's
`RandomForestClassifier`. It was trained with 100 decision trees
(`n_estimators=100`) and a fixed random seed (`random_state=42`) for
reproducibility. The model performs binary classification, predicting
whether an individual's annual income is above or below $50,000.
It was developed by Emy Kirugo as part of a machine learning pipeline
deployment project using FastAPI, July 2026.

## Intended Use
This model is intended for educational and demonstration purposes as
part of a machine learning deployment pipeline project. It predicts
whether an individual's income exceeds $50,000 per year based on
demographic and employment attributes from U.S. Census data. It is
not intended for use in real-world decision-making processes such as
lending, hiring, insurance, or any context that could materially
affect a person's life, given known limitations and biases in the
underlying training data (see Ethical Considerations).

## Training Data
The model was trained on the UCI Census Income dataset (`census.csv`),
which contains 32,561 rows of U.S. Census data. The target label is
`salary`, a binary variable indicating whether an individual earns
`<=50K` or `>50K` annually. Eight categorical features (workclass,
education, marital-status, occupation, relationship, race, sex, and
native-country) were one-hot encoded, and the label was binarized,
using the `process_data` function. The dataset was split into 80%
training data and 20% test data (`random_state=42`). Some columns in
the raw data contain `"?"` to represent missing values, which were
not explicitly removed or imputed prior to training.

## Evaluation Data
The model was evaluated on the 20% test split (approximately 6,513
rows) held out from the same `census.csv` dataset described above.
The same preprocessing steps (one-hot encoding and label binarization)
were applied to the test data, reusing the encoder and label binarizer
fitted on the training data to ensure consistent transformations.

## Metrics
The model was evaluated using precision, recall, and F1 score (F-beta
with beta=1), computed via scikit-learn's `precision_score`,
`recall_score`, and `fbeta_score` functions. On the held-out test set,
the model achieved:

- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

Performance was also computed on slices of the test data for each
value of every categorical feature (workclass, education,
marital-status, occupation, relationship, race, sex, and
native-country), using the `performance_on_categorical_slice`
function. This slice-level output is saved in `slice_output.txt` and
shows that performance varies meaningfully across subgroups — for
example, some categories with very small sample sizes (such as
`workclass: Without-pay, Count: 4`) show perfect or highly volatile
scores due to the small number of examples, while larger categories
like `workclass: Private` more closely track the overall model
performance.

## Ethical Considerations
This model was trained on U.S. Census data that includes sensitive
demographic attributes such as race, sex, and native-country. Prior
research on this dataset has documented that income prediction models
trained on it can reflect and reproduce historical biases and
disparities present in the underlying data (for example, differences
in predicted income across race and sex). The slice-based performance
analysis in this project (see `slice_output.txt`) shows that model
performance is not uniform across categories, and this includes
protected attributes. Some subgroups also have very small sample
sizes in the test data, which makes their performance metrics
unreliable and easily misread as either better or worse than they
really are. Because of these risks, this model should not be used to
make or inform real decisions about individuals, and any use of race,
sex, or national origin as predictive features warrants careful
scrutiny regardless of a model's aggregate accuracy.

## Caveats and Recommendations
- The model uses default hyperparameters for `RandomForestClassifier`
  aside from `n_estimators=100`; no hyperparameter tuning or cross-
  validation was performed. Grid search or randomized search could
  likely improve performance.
- Missing values represented as `"?"` in the raw data were not
  explicitly cleaned, imputed, or removed before training, which may
  affect model quality for the affected columns.
- Some categorical slices have very small sample sizes (single or low
  double digits), making their individual metrics statistically
  unreliable; conclusions should not be drawn from these slices in
  isolation.
- This model was trained on a single train/test split rather than
  k-fold cross-validation, so reported metrics may vary somewhat with
  a different random seed or split.
- Before any real-world use, this model would require a fairness
  audit, retraining with bias-mitigation techniques, and validation
  on more recent, representative data, since the underlying Census
  data may not reflect current demographic or economic conditions.