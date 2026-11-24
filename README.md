# Flight Delay Predictor

Predict whether a flight will arrive more than 15 minutes late, using only information known before departure.

## STAR
- **Situation:** About 1 in 4 flights from New York arrives late. Knowing the risk before takeoff is useful, but delay data is easy to misuse because departure delay leaks the answer.
- **Task:** Build an honest pre-departure model for late arrival (arrival delay above 15 minutes) and measure it against a majority-class baseline.
- **Action:** Used the nycflights13 data (336,776 flights from EWR, JFK and LGA in 2013, 327,346 after removing missing values, 23.7% late). Deliberately **did not use departure delay** as a feature. Used month, day of week, scheduled departure hour, carrier, origin, destination and distance on a 120,000 flight sample (80/20 stratified split, 24,000 test flights). Compared Logistic Regression and Gradient Boosting, tested three decision thresholds and checked permutation importance.
- **Result:** Gradient Boosting reached **AUC 0.722** and **77.6% accuracy** against 76.2% for always predicting "on time". Lowering the threshold to 0.2 catches **74.5% of late flights** (precision 34.9%).

## Results (test set, 24,000 flights)
| Model | Accuracy | AUC |
|---|---|---|
| Always predict on time (baseline) | 76.16% | - |
| Logistic Regression | 76.16% | 0.638 |
| Gradient Boosting | **77.63%** | **0.722** |

Gradient Boosting at different thresholds (late class):
| Threshold | Precision | Recall |
|---|---|---|
| 0.5 | 61.3% | 16.7% |
| 0.3 | 44.4% | 49.5% |
| 0.2 | 34.9% | 74.5% |

## Key insights
- Late share rises through the day, from 9.9% for 5 AM departures to 36.6% for 9 PM departures, because delays build up during the day.
- Permutation importance (AUC drop): departure hour 0.109, month 0.081, day of week 0.054, carrier 0.040, distance 0.013, destination 0.008, origin 0.008.
- Accuracy barely beats the baseline because 76% of flights are on time, so AUC and recall by threshold tell the real story.

## Limitations
- Delay depends a lot on weather and air traffic, which are not in the features, so the ceiling is low without them.
- Random split, not a time split, and only one year of data.
- Trained on a 120,000 flight sample for speed.

## How to run
1. Open `flight_delay_predictor.ipynb` in Google Colab.
2. Runtime > Run all (it installs `nycflights13` in the first cell).
3. The first data cell should print "real data loaded".

## Files
- `flight_delay_predictor.ipynb` - executed notebook with outputs
- `requirements.txt`

## Author
Akshat Kesharwani
- Portfolio: https://akshatkesharwani-info.github.io/
- GitHub: https://github.com/akshatkesharwani-info
