![If your RAG model retrieves documents from the Malicious Misinformer, you probably need a media background check](https://github.com/MichSchli/MediaBackgroundChecks/blob/main/Background_check_example.png)

# Media Background Checks

This repository maintains the dataset and models described in our paper [Generating Media Background Checks for Automated Source Critical Reasoning](todo). We propose to generate Media Background Checks (MBCs) that summarise indicators of trustworthiness and tendency for media sources. MBCs can be used, either by humans or by retrieval-augmented models, to determine which documents can be relied on for further reasoning, and to craft reliable narratives based on untrustworthy evidence.

## Dataset Structure

The dataset is structured as follows. First, the media background checks themselves can be found in *data/mbcs*. The division into splits can be found in *data/splits*. Each split file (train, dev, test) contains, separated by newlines, the filenames of the background checks associated with that split. The GPT-3.5 generated atomic facts we used for FactScore-style evaluation in the paper can be found in *data/splits/dev_facts* and *data/splits/test_facts*, respectively. Finally, *data/* also contains the controversial and misinformative question-answer pairs we constructed for human evaluation.

## Evaluation

Evaluation scripts with atomic facts (i.e., FactScore) and traditional metrics are found respectively in *eval_with_atomics.py* and *eval_with_metrics.py*. To run the scripts, please use:

```
python eval_with_atomics.py --predictions_folder your_output_folder --dataset_file data/splits/dev.tsv --fact_folder data/splits/dev_facts
python eval_with_metrics.py --predictions_folder your_output_folder --dataset_file data/splits/dev.tsv --reference_folder data/mbcs
```

## Baseline

Our baseline can be run as follows:

```
python generate_media_background_checks.py --predictions_folder your_output_folder --dataset_file data/splits/dev.tsv
```

## Citation

If you used our dataset or code, please cite our paper as:


```
@misc{schlichtkrull2024mediabackgroundchecks,
      title={Generating Media Background Checks for Automated Source Critical Reasoning}, 
      author={Michael Schlichtkrull},
      year={2024},
      eprint={2409.00781},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.00781}, 
}
```

## License

All credit for the creation of the background checks included in this dataset goes to the Media Bias / Fact Check team. Assessments of credibility and bias can change over time, and if you are trying to evaluate the credibility of a media outlet it is important to use the most recent information. For up-to-date versions of each background check, please visit https://mediabiasfactcheck.com/.

<br/>
<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</p>
