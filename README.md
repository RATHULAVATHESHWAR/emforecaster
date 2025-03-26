# EMForecaster
## Overview
This is the official repository for the paper:

>[Xavier Mootoo, Hina Tabassum, Luca Chiaraviglio. EMForecaster: A Deep Learning Framework for Time Series Forecasting in Wireless Networks with Distribution-Free Uncertainty Quantification (IEEE Transactions on Network Science and Engineering, under review).](https://xaviermootoo.com/s/emforecaster.pdf)

## Description
EMForecaster is a novel deep learning architecture specialized for time series forecasting, benchmarked primarily on electromagnetic field (EMF) exposure forecasting. EMForecaster also includes a conformal prediction pipeline for uncertainty quantification, along with a trade-off score metric which we propose a unified measure of model performance to balance width of prediction intervals (minimizing) and empirical coverage (maximizing).


## Installation
### Dependencies
- Python $\geq$ 3.10

### Using conda
```bash
# Create and activate conda environment
conda create -n emforecaster python=3.10
conda activate emforecaster

# Install requirements
pip install -r requirements.txt
pip install -e .
```

### Using pip
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -e .
```

## Data
Data is proprietary and provided by primarily by Luca Chiaraviglio, please contact him for access.

## Usage

### Running Experiments
Execute the main script with the desired model:
```python
python main.py <model>
```

Results are saved in the `logs` folder if using offline logging. To run custom experiments, run `emforecaster/jobs/exp/<model>/args.yaml` for each model, with respect to the classes in [`config.py`](/emforecaster/config/config.py).


### Hyperparameter Tuning
Hyperparameter tuning is also available, by modifying `emforecaster/jobs/exp/<model>/ablation.yaml`, which performs grid search on a set of parameters. To run hyperparameter tuning jobs use:
```python
python /tuning/tune.py <model>
```

### Neptune Tracking
This project allows for online experimental logging via Neptune.ai, which is free for researchers and students.

1. Create a Neptune.ai account and API token
2. Set your Neptune API token as an environment variable:
```bash
export NEPTUNE_API_TOKEN='your-neptune-api-token'
```
3. Set `neptune: True` in your `args.yaml` under the `exp` category. See other parameters such as `run_id` and `exp_id`.

The [`get_results.py`](/emforecaster/analysis/neptune/get_results.py) reads your `<model>.yaml` file in the [`emforecaster/analysis/neptune/ablations`](/emforecaster/analysis/neptune/ablations) directory, which iterates through all model runs (e.g., from a hyperparameter run) to display the best result with respect to a `deciding_metric` (e.g., MSE). All model runs and their model configurations are displayed and saved as `results.csv`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations
If you use this code in your research or work, please cite our paper:
```bibtex
@article{mootoo2025emforecaster,
    title     = {EMForecaster: A Deep Learning Framework for Time Series Forecasting in Wireless Networks with Distribution-Free Uncertainty Quantification},
    author    = {Mootoo, Xavier and Chiaraviglio, Luca and Tabassum, Hina},
    year      = {2025},
}
```

## Contact
For queries, please contact the corresponding author through: `xmootoo at gmail dot com`.

## Acknowledgments
Xavier Mootoo is supported by Canada Graduate Scholarships - Master's (CGS-M) funded by the [Natural Sciences and Engineering Research Council](https://www.nserc-crsng.gc.ca/index_eng.asp) (NSERC) of Canada, the Vector Scholarship in Artificial Intelligence, provided through the [Vector Institute](https://vectorinstitute.ai/), Canada, and the Ontario Graduate Scholarship (OGS) granted by the provincial government of Ontario, Canada. 

We extend our gratitude to [Commune AI](https://communeai.org/) for generously providing the computational resources needed to carry out our experiments, in particular, we thank Luca Vivona ([@LVivona](https://github.com/LVivona)) and Sal Vivona ([@salvivona](https://github.com/salvivona)).