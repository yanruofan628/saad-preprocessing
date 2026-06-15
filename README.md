# SAAD preprocessing and validation code

This repository contains the curated analysis code used for the **SAAD** dataset manuscript  
(**S**timulus-driven **A**uditory **A**ttention **D**ichotic): an open 128-channel EEG resource for  
instruction-free dichotic listening with naturalistic sounds.

The code is organized by analysis stage rather than by the original working folders, so the public  
repository is easier to audit and reuse. Scripts assume you have downloaded the dataset from Zenodo  
and configured local paths (see **Data assumptions** below).

## Dataset

The dataset (raw EEG, preprocessed epochs, dichotic stimuli, and post-experiment ratings) is hosted on Zenodo:

- DOI: [10.5281/zenodo.20557225](https://doi.org/10.5281/zenodo.20557225)
- Record: [https://zenodo.org/records/20557225](https://zenodo.org/records/20557225)

Each participant contributes 480 dichotic trials (three sessions × 160 trials). Processed EEG arrays  
are stored as 6 s epochs (250 Hz, 128 channels) with dichotic onset at sample index 750  
(3 s pre-stimulus baseline, 2 s dichotic stimulation, 1 s post-stimulus silence).

## Directory layout

```
preprocessing/
  preprocess_eeg.py                 # preprocessing pipeline

technical_validation/
  signal_quality/
    plot_preproc_interp_and_ica_violins.py   # Interpolated-channel and rejected-IC summaries
    plot_subject_ica_components.py   #plot ica components
    plot_topomap.py   # Group PSD and band topographies (baseline vs stimulus)
  behavior/
    plot_rt_excluded_violin.py            # Excluded trials (RT > 5000 ms)
    plot_rt_session.py                    # RT distributions across sessions
    category_attraction.py                # Caculate selecting rate for each category
    plot_pair_aggregate_glm_forest.py     # Pair-level acoustic-feature GLM
  aad_baseline/
    eegnet_2s_cv5.py                      # EEGNet baseline on 0–2 s stimulation window
    Mirror- and category-held-out CV.py   # Mirror- and category-held-out CV extensions
    pair_consistent_splits.py             # Mirror-constrained and subcategory 6-fold split helpers
    plot_bacc_three_strategies_boxplot.py # Three CV schemes (manuscript Fig. signal-g)

```

If your clone uses a flat layout, the script names above are the authoritative entry points; reorganizing  
into these folders is recommended before publication but not required for execution.

## Data assumptions

### Zenodo release layout

The scripts that package or validate the public dataset expect a local root directory (conventionally  
`SAAD_dataset/`) with this structure:

```
SAAD_dataset/
  README.md
  dataset_description.json
  participants.tsv
  CHANGES.txt
  raw_eeg/
    sub-XX/
      ses-01/ ... ses-03/
        sub-XX_ses-YY_task-SAAD_run-01_eeg.mff/
  processed/
    sub-XX/
      sub-XX_preprocessed_eeg.npy      # shape (480, 128, 1500)
      sub-XX_preprocessed_labels.tsv   # Label: 0=left, 1=right
      sub-XX_preprocessed_rts.tsv      # RT_ms
      sub-XX_preprocessed_meta.json
  stimuli/
    wav/ ... pairs/ ... trial_stimulus_index.tsv
  ratings/
    ratings_scale_description.tsv
    participant_ratings.tsv
```

See the dataset `README.md` on Zenodo for the full tree and field definitions.

### Raw acquisition files (preprocessing only)

`preprocess_eeg.py` additionally expects locally stored EGI `.mff` recordings and E-Prime benchmark  
`.txt` logs. Paths are configured in `subject_mff_benchmark_config.py` and in script-level  
`BASE_DATA_DIRS` / `SAAD_ROOT` variables. Before running, point these to your local copies of the  
raw `.mff` folders and benchmark logs.

Several scripts use hard-coded Windows paths or `/path/to/...`-style placeholders. Set environment  
variables or edit the path constants at the top of each script before running.

## Recommended workflow

Reproduce the manuscript pipeline in this order:

1. **Preprocess raw EEG**  
   Run `preprocessing/preprocess_eeg.py --all-subjects --full-export --output-root <deriv_root>`.  
   This applies GSN montage, PyPREP bad-channel interpolation, 0.1–40 Hz FIR band-pass, 50 Hz notch,  
   common-average re-reference, Infomax ICA with ICLabel artifact rejection, and exports 6 s trial epochs  
   with aligned labels and RTs.

2. **Refresh behavioral tables (if re-parsing benchmark logs)**  
   Run `release/refresh_labels_rts_from_benchmark.py` to update `preprocessed_labels.tsv` and  
   `preprocessed_rts.tsv` in the Zenodo-style release tree.

3. **Import subjective ratings (optional, for packaging)**  
   Run `release/import_ratings_to_saad.py` with your local ratings spreadsheet to populate  
   `ratings/participant_ratings.tsv`.

4. **Anonymize and audit before upload**  
   Run `release/anonymize_saad_dataset.py`, then `release/audit_saad_dataset.py` and  
   `release/audit_data_records_match.py` to verify the public folder tree and absence of  
   participant identifiers.

5. **Technical validation — signal quality**  
   Run `technical_validation/signal_quality/export_preproc_violin_for_ngplot.py` and  
   `plot_preproc_interp_and_ica_violins.py` for interpolated-channel and rejected-IC summaries;  
   run `plot_topomap_group_delta_and_all.py` for group PSD and band topographies.

6. **Technical validation — behavior**  
   Run `technical_validation/behavior/extract_all_subjects_rt.py`, then the RT/selection plotting scripts  
   listed above for excluded-trial counts, session RT distributions, category selection rates, and  
   pair-level acoustic-feature models.

7. **Technical validation — baseline decoding**  
   - Random split: `technical_validation/aad_baseline/eegnet_2s_cv5.py`  
   - Mirror-constrained split: same script with `--pair-consistent`  
   - Category-held-out split: `eegnet_classify_trials_5s_1s_cv5.py --subcategory-6fold`  
   Summarize the three schemes with `plot_bacc_three_strategies_boxplot.py`.

Generated figures, logs, trained model weights, raw `.mff` recordings, and the Zenodo dataset bundle  
itself are **not** included in this code repository.

## Dependencies

Install Python **3.12+** and the packages listed in `requirements.txt`:

- `mne` (validated with v1.10.2)
- `mne-icalabel`
- `pyprep` (v0.4.3)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `torch` (for EEGNet baseline decoding)
- `scikit-learn`, `statsmodels` (behavioral GLM analyses)
- `mffpy` (raw `.mff` I/O during preprocessing)

Exact versions were not fully pinned during the original analysis. For archival reuse, consider  
exporting the working Conda environment alongside this repository.

## Usage notes

- Trial labels are **post-stimulus self-reports** of the more attention-capturing side, not  
  experimenter-cued attended-ear ground truth.
- In processed arrays, **time 0 s** is dichotic-stimulus onset (`t = -3` to `0` baseline;  
  `0` to `2` stimulation; `2` to `3` post-stimulus silence before the motor response).
- Load released epochs with NumPy; load `.mff` raw files with MNE-Python as described in the manuscript.

## Citation

If you use this code or the SAAD dataset, please cite:

1. The Zenodo record: [10.5281/zenodo.20557225](https://doi.org/10.5281/zenodo.20557225)
2. The accompanying *Scientific Data* Data Descriptor (citation to be added upon publication)

## License

Add your chosen license here (e.g. MIT) before public release.
