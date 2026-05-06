# Project Summary

## 1. Project goal
The project aims to predict the **difficulty level of a piano piece** from symbolic score information extracted from digital sheet music, and later extend that prediction layer with a recommendation component.

At this stage, the core of the project is clearly the **difficulty prediction task**.

---

## 2. Dataset choices and current status
### Initial practical choice
The working prototype was built on **Mikrokosmos-difficulty** because it was immediately available, structured, and usable end-to-end.

Why it was chosen:
- MusicXML files were available locally and could be parsed automatically;
- a difficulty label existed through `henle_difficulty`;
- it allowed the project to move forward without waiting for a harder-to-access dataset.

### Data cleaning and filtering
The raw metadata file contained **153 rows**.

The filtering logic kept only the relevant solo piano pieces:
- 6 entries were excluded because they were **4 hands** pieces;
- the final usable set contained **147 solo pieces** with matching MusicXML files.

### Current target construction
The original `henle_difficulty` labels were grouped into 3 classes:
- `beginner` for levels 1 to 3
- `intermediate` for levels 4 to 6
- `advanced` for levels 7+

Final class distribution:
- `beginner`: 92
- `intermediate`: 45
- `advanced`: 10

### Important limitation
The dataset is clean enough for a prototype, but it remains limited:
- small size;
- very small advanced class;
- stylistically narrow corpus;
- all pieces come from Béla Bartók’s *Mikrokosmos*.

So the dataset is **good enough for a proof of concept**, but not broad enough for strong generalization claims.

---

## 3. Why MusicXML matters
The project does not rely only on a metadata CSV.

The **MusicXML files** are central because they contain the symbolic content of the score in a machine-readable form. They make it possible to extract features such as:
- note counts;
- rests;
- durations;
- chords;
- pitch ranges;
- melodic intervals;
- tempo indications;
- key signatures;
- time signature changes.

Without MusicXML, the project would not have access to the actual musical structure needed for feature engineering.

---

## 4. Final feature set currently retained
The current prototype keeps a feature set focused on symbolic difficulty proxies.

### Global volume and density
- `notes_total`
- `notes_played`
- `rests`
- `measures`
- `notes_per_measure`

### Chords and texture
- `chord_notes`
- `chord_ratio`

### Pitch and melodic motion
- `pitch_span`
- `unique_pitch_count`
- `avg_pitch_interval`
- `max_pitch_interval`

### Rhythmic and temporal structure
- `avg_duration`
- `duration_std`
- `duration_cv_proxy`
- `rhythmic_variety`
- `tempo_mean`
- `notes_per_second_proxy`
- `rest_ratio`

### Normalized symbolic density
- `notes_per_pitch_class`
- `notes_per_measure_per_pitch_class`

### Harmonic / notation complexity proxies
- `accidental_ratio`
- `key_signature_complexity`
- `time_signature_changes`

### Corpus context
- `book_code`

These features were kept because they provided a reasonable balance between interpretability and empirical usefulness.

---

## 5. Feature engineering choices and lessons learned
A lot of the work focused on testing whether additional symbolic features actually improved the model.

### Features that were explored but not retained in the main training pipeline
The following exploratory features were implemented or tested at different points:
- `pitch_std`
- `large_leap_ratio`
- `avg_chord_cluster_size`
- `span_per_unique_pitch`
- `notes_per_second_per_pitch_class`
- `tempo_duration_interaction`

### Additional feature families tested and rolled back
Several extra feature families were tested and then rolled back because they did not improve performance:
- variability of note density across measures;
- larger chord-cluster oriented features;
- chromatic step ratio;
- melodic direction-change features;
- interval variability features;
- extreme register usage features.

### Main lesson
A strong conclusion from the experiments is:

> **adding more musically plausible features does not automatically improve the model.**

On this dataset, several intuitive additions made the model worse or left it unchanged. The limitation seems to come more from the dataset size and composition than from a lack of raw feature count.

---

## 6. Models tested and current best model
Three baselines are now used in the project so the comparison stays broad enough for Assignment 3:

### Logistic Regression
Used as an interpretable linear baseline with scaling and class balancing.

### SVM (RBF)
Used as a kernel baseline on standardized symbolic features to test a nonlinear margin-based classifier.

### Random Forest
Used as a nonlinear baseline better suited to interactions between handcrafted symbolic features.

### Current metrics
After the current best retained feature set:

- **Logistic Regression**
  - holdout accuracy: 0.8667
  - holdout macro F1: 0.5910
  - cross-validation accuracy mean: 0.7866
  - cross-validation accuracy std: 0.0367
  - cross-validation macro F1 mean: 0.6576
  - cross-validation macro F1 std: 0.1186

- **SVM (RBF)**
  - holdout accuracy: 0.7333
  - holdout macro F1: 0.5051
  - cross-validation accuracy mean: 0.7783
  - cross-validation accuracy std: 0.0980
  - cross-validation macro F1 mean: 0.5601
  - cross-validation macro F1 std: 0.0957

- **Random Forest**
  - holdout accuracy: 0.8667
  - holdout macro F1: 0.7481
  - cross-validation accuracy mean: 0.8033
  - cross-validation accuracy std: 0.0931
  - cross-validation macro F1 mean: 0.6855
  - cross-validation macro F1 std: 0.1465

### Current best choice
The best practical baseline at the moment is still the **Random Forest**.

It performs better on holdout macro F1 and remains slightly ahead on cross-validation macro F1. The **SVM (RBF)** satisfies the third-model comparison requirement, but it is clearly weaker than the Random Forest on the current corpus.

### Optuna follow-up on the best baseline
To push the project a bit further than manual baseline settings, an **Optuna** tuning pass was added for the Random Forest.

This tuning runs **as a separate script** and maximizes **cross-validation macro F1** on the full `X, y` dataset. So it should be read as a **follow-up hyperparameter-search experiment**, not yet as the main baseline row used in the comparison table above.

After **20 trials**, the best configuration found was:
- `n_estimators = 400`
- `max_depth = 10`
- `min_samples_leaf = 4`
- `max_features = log2`

The best cross-validation macro F1 reached about **0.7253**.

This is useful for Assignment 3 because it shows the project goes beyond comparing three baselines and also attempts principled hyperparameter optimization on the strongest candidate. But the score should not be over-read as a strict apples-to-apples replacement for the baseline table above, since the tuning protocol is separate.

---

## 7. Model analysis and error patterns
The current Random Forest performs well overall, but the analysis showed a clear pattern:
- `beginner` pieces are recognized well;
- `intermediate` pieces are reasonably handled;
- `advanced` pieces remain the weakest class.

This is consistent with the data distribution, since the advanced class is very small.

Feature-importance analysis also suggested that the model relies strongly on:
- `measures`
- `notes_total`
- `notes_played`
- `notes_per_pitch_class`
- `accidental_ratio`
- `rhythmic_variety`
- `book_code`
- `duration_cv_proxy`

This confirms that the feature set carries signal, but it also shows that the model partly benefits from the pedagogical structure of the Mikrokosmos corpus itself.

---

## 8. PCA experiment
A PCA-based dimensionality reduction experiment was run to test whether compressing the feature space would improve generalization.

### Outcome
- some PCA settings improved cross-validation macro F1 for Logistic Regression;
- however, those same settings degraded the holdout performance;
- PCA did not beat the best non-PCA Random Forest baseline.

### Conclusion
PCA was therefore kept as an **experiment documented in the process**, but **not adopted in the final pipeline**.

---

## 9. CIPI preparation
Later in the project, it became possible that access to **CIPI** might be granted.

To prepare for that without breaking the current prototype, the loading pipeline was made **dataset-selectable via environment variables**:
- `PIANO_DATASET` controls which dataset is selected;
- default is still `mikrokosmos`;
- `CIPI_DIR` can be used once the CIPI dataset is available locally.

The current behavior is safe:
- Mikrokosmos remains the default working dataset;
- an unknown dataset name fails clearly;
- selecting `PIANO_DATASET=cipi` without an available dataset fails clearly with a `FileNotFoundError`.

This means the project is now better prepared for integrating CIPI later, without silently falling back to the wrong dataset.

---

## 10. Deliverables completed so far
### Assignment 1
Assignment 1 was updated so that the project description reflects the actual prototype path based on Mikrokosmos as the first working dataset.

### Assignment 2
Assignment 2 was drafted to explain:
- the source dataset;
- the filtering and cleaning process;
- the target construction;
- the full feature engineering logic;
- the rejected and retained features;
- the PCA experiment;
- the current dataset limitations.

---

## 11. Current state of the project
At the moment, the project has:
- a working end-to-end symbolic piano difficulty pipeline;
- tested feature engineering choices;
- a stable best baseline;
- documented rejected experiments;
- a dataset-switching preparation layer for future CIPI integration.

The project is therefore in a good prototype state:
- technically functional;
- empirically explored;
- documented enough to justify the current choices;
- ready for a future dataset upgrade.

---

## 12. Main takeaway
The most important conclusion so far is that the current bottleneck is **not simply “we need more features.”**

The prototype already has a solid symbolic feature set and a working model. The real limitations now are mostly:
- dataset size;
- class imbalance;
- stylistic narrowness of the corpus.

That is why the next major gain will likely come more from **integrating a richer dataset such as CIPI** than from endlessly adding handcrafted features to Mikrokosmos.
