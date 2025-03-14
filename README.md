# Study on forecasting weekly time series with an N-BEATS foundation model
Implementation of pre-training, forecasting and evaluating an N-BEATS foundation model and assessing the relationship to data set similarity, diversity and time series features.

## Framework
(1) First, we create the data basis of time series by concatenating weekly publicly available data sets.
(2) Then, we calculate ten time series features for each series.
(3) We create increasingly diverse (in regard to the features) so called original source data sets. 
(4) From each of them, we create a source and multiple target data sets, which are increasingly similar (in regard to the features) to the source.
(5) We pre-train an N-BEATS ensemble on each source and do zero-shot forecasts on the respective target data sets. The source performance is also calculated.
(6) In the evaluation, we investigate the relationships between source and target performances, similarity and diversity and the features.


## Getting started
We need two different poetry environments. Therefore, an installation of poetry (https://python-poetry.org/docs/#installing-with-the-official-installer) and pyenv (https://github.com/pyenv/pyenv#installation) is needed. 
The "main environment" can be created with the "pyproject.toml" by running "poetry install". 
For the second environment go to "src/experiments/data-prep" and run "poetry install" again.
Set the path to the "data-prep environment" in the config file: "hydra_configs/config.yaml".
Create a "data" folder outside of "src" and an "M5" folder inside.
Get the M5 data set "sales_train_validation.csv" from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data and put them in the folder "data/M5".

## Usage
We refer to one run as running the framework described above once with a specific config file.
In the paper, we report the results over several runs.
After filling out the config file, run "src/experiments/run.py". 
The results will be saved in the respective folder in "/data".
For evaluation of single and multiple runs, use the jupyter notebooks in the "evaluation" folder. 

## Scripts
For each step of the framework, we use the following scripts:
(1) "src/experiments/data_creation/create_concat.py"
(2) "src/experiments/data_creation/selected_tsfresh_features.py"
(3) "src/experiments/data_creation/create_diverse_sources.py"
(4) "src/experiments/data_creation/create_sources_and_targets.py"
(5) "src/experiments/transfer_learning/run_tl.py"
(6) "src/experiments/evaluation/1_evaluate_single_runs.ipynb", "src/experiments/evaluation/2_evaluate_over_runs.ipynb"

## References
[1] Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., ... & Wang, Y. (2020). Gluonts: Probabilistic and neural time series modeling in python. Journal of Machine Learning Research, 21(116), 1-6.

[2] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & Van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.

[3] Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests (tsfresh–a python package). Neurocomputing, 307, 72-77.

[4] Bradley, P. S., Bennett, K. P., & Demiriz, A. (2000). Constrained k-means clustering. Microsoft Research, Redmond, 20(0), 0.

## Dataset References
[D1] Maggie, Oren Anava, Vitaly Kuznetsov, and Will Cukierski. Web Traffic Time Series Forecasting. https://kaggle.com/competitions/web-traffic-time-series-forecasting, 2017. Kaggle.

[D2] Crone, S. (2008). NN5 Forecasting Competition. http://www.neural-forecasting-competition.com/index.htm 

[D3] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1), 54-74.

[D4] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4), 1346-1364.

[D5] Lai, G., Chang, W. C., Yang, Y., & Liu, H. (2018, June). Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval (pp. 95-104).






