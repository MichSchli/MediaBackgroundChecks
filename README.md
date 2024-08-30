# Media Background Checks

This repository maintains the dataset and models described in our paper TODO.



## Dataset Structure


## Evaluation

Evaluation scripts with atomic facts (i.e., FactScore) and traditional metrics are found respectively in *eval_with_atomics.py* and *eval_with_metrics.py*. To run the scripts, please use:

```
python eval_with_atomics.py --predictions_folder your_predictions --dataset_file data/dataset/dev.tsv --atomic_fact_folder data/dataset/dev_facts/atomic_facts
python eval_with_metrics.py --predictions_folder your_predictions --dataset_file data/dataset/dev.tsv
```

## Baseline

Our baseline can be run as follows:

```
python generate_media_background_checks.py --output_folder your_predictions --dataset_file data/dataset/dev.tsv
```

## License

All credit for the creation of the background checks included in this dataset goes to [Media Bias / Fact Check](https://mediabiasfactcheck.com/). For up-to-date versions of each background check, please visit their website.

<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</p>
