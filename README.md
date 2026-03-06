# Public Transport Delay Prediction

This project predicts **public transport delays** using timetable, weather, traffic, holiday, and city-event information.

## Dataset inspected
- Rows: **2,000**
- Columns: **24**
- Missing values: mainly in `event_type` (missing usually means no event)
- Useful targets available:
  - `delayed` -> classification target
  - `actual_arrival_delay_min` -> regression target

## Important modeling note
To keep the project realistic and avoid leakage:

- Do **not** use `actual_departure_delay_min`
- Do **not** use `actual_arrival_delay_min` when predicting `delayed`
- Do **not** use `delayed` when predicting `actual_arrival_delay_min`

These columns describe the actual outcome itself.

## Recommended project scope
### Main project
Predict `actual_arrival_delay_min` from:
- route and transport information
- schedule time
- weather
- temperature / precipitation / wind
- traffic congestion
- holiday / peak hour / weekday / season
- event information

### Bonus extension
Predict `delayed` as a classification problem.

## Folder contents
- `notebook.ipynb` -> EDA, plots, markdown explanations
- `train.py` -> trains regression model and saves best pipeline
- `predict.py` -> loads saved model and predicts delay for sample/new data
- `requirements.txt` -> dependencies
- `public_transport_delays.csv` -> dataset copy

## How to run
```bash
pip install -r requirements.txt
python train.py
python predict.py
```

## Output
Training saves:
- `best_delay_model.joblib`
- `metrics.json`

## Suggested workflow
1. Open `notebook.ipynb`
2. Inspect distributions and missing values
3. Confirm target = `actual_arrival_delay_min`
4. Review leakage discussion
5. Run `train.py`
6. Check `metrics.json`
7. Run `predict.py`
