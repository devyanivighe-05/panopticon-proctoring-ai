# Project Panopticon: Intelligent Exam Proctoring

Time-series machine learning that flags cheating only when highly certain, to protect innocent students.
Built as the capstone of my Artificial Intelligence Internship at Internship Studio (EduGuard AI, Machine Learning Division).

## Problem
Online proctoring tracks eye gaze, background audio and computer logs, but flags innocent students for sneezing,
reading a question aloud or glancing at the keyboard. A false accusation can damage an academic career, so the
system must flag a student only when it is highly certain (90%+).

Challenges:
- Video telemetry (every 1 second) and system events (only when something happens) are asynchronous
- Dropped video leaves gaps of missing values
- A 1-second glance is not cheating
- Only 77 of ~10,000 seconds are cheating, so accuracy is misleading

## Approach
- **Align:** `pd.merge_asof` (backward, 1-second tolerance) attaches each system event to the nearest video second
- **Clean:** forward-fill eye gaze, linearly interpolate audio, fill missing `tab_switches` / `is_cheating` with 0 (732 dropped seconds recovered)
- **Engineer:** 10-second rolling mean of gaze and rolling max of audio, so only sustained behaviour counts
- **Model:** `RandomForestClassifier` with `class_weight='balanced'`, 80/20 stratified split, `random_state=42`
- **Decide:** `predict_proba` with a custom 0.90 threshold instead of 0.50, plus a Precision-Recall curve
- **Validate:** 5-fold stratified cross-validation over all 9,991 rows

## Results (cross-validated)
| Threshold | Precision | Recall | Innocent students flagged |
|-----------|-----------|--------|---------------------------|
| 0.50      | 0.86      | 0.88   | 11                        |
| 0.90      | 0.94      | 0.38   | 2                         |

Top features: `tab_switches`, `audio_db`, `eye_gaze_angle`, 10-second gaze rolling mean.

**Trade-off:** the strict threshold misses some cheaters on purpose. Flags should go to a human reviewer and
never trigger automatic penalties. Recommendation: use as a human-in-the-loop review tool at the 0.90 threshold.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter / Google Colab

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Panopticon_Proctoring_AI.ipynb
```
The input files `video_telemetry.csv` and `system_events.csv` are not included in this repository. Place them next to the notebook before running (in Google Colab, upload them using the folder icon on the left).

## Links
- [Walkthrough video](https://drive.google.com/file/d/1EpJIcCkBan0Mhb_dhe45v_EpdhYsvOIE/view?usp=sharing)
- [Project Files](https://drive.google.com/drive/folders/1mRkeg-IjWrX_-ieIqYsqzsmc2vEGq51J?usp=sharing)

## Author
Devyani Vighe
