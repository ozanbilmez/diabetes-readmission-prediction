# Data

## diabetic_data.csv and IDs_mapping.csv (included)

Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- Licence: CC BY 4.0. Both files are included unchanged.
- `diabetic_data.csv`: 101,766 inpatient encounters of 71,518 diabetic patients, 50 columns.
- `IDs_mapping.csv`: what the codes in `admission_type_id`, `discharge_disposition_id` and `admission_source_id` mean (e.g. discharge code 22 is a transfer to a rehab facility).
- Missing values are written as `?`. In `max_glu_serum` and `A1Cresult`, `None` means the test was not done, so it is kept as its own category.
- Target used here: `readmitted == "<30"`, i.e. readmitted within 30 days (11.2% of encounters).

## hospital_readmissions.csv (not included, download it)

"Predicting Hospital Readmissions" on Kaggle: https://www.kaggle.com/datasets/dubradave/hospital-readmissions

The licence on the dataset page is "Other (specified in description)", but the description does not state any terms, so the file is not redistributed here. Download it and save it as `data/hospital_readmissions.csv`. Without it, `run_experiment.py` runs on the UCI data only and the Kaggle test is skipped.

- 25,000 encounters, 17 columns. Diagnoses are already grouped into categories.
- The page describes it as ten years of patient records with the same kind of fields as the UCI data, so it looks like a simplified extract of it.
- `readmitted` is a plain yes/no with no time window (47% positive), which makes it a different and more balanced task than the 30-day target above.
