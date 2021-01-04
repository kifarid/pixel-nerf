# pixelNeRF: Neural Radiance Fields from One or Few Images

Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa<br>
UC Berkeley

![Teaser](https://raw.github.com/sxyu/pixel-nerf/master/readme-img/paper_teaser.jpg)

arXiv: http://arxiv.org/abs/2012.02190

This is a *temporary* code repository for our paper, pixelNeRF, pending final release.
The official repository shall be <https://github.com/sxyu/pixel-nerf>.

# Getting the data

- For the main ShapeNet experiments, we use the ShapeNet 64x64 dataset from NMR
https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
(Hosted by DVR authors) 
    - For novel-category generalization experiment, a custom split is needed.
      Download the following script:
      https://drive.google.com/file/d/1Uxf0GguAUTSFIDD_7zuPbxk1C9WgXjce/view?usp=sharing
      place the said file under `NMR_Dataset` and run `python genlist.py` in the said directory.
      This generates train/val/test lists for the experiment. Note for evaluation performance reasons,
      test is only 1/4 of the unseen categories.

- The remaining datasets may be found in
https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing
    - Custom two-chair `multi_chair_*.zip`
    - DTU (4x downsampled, rescaled) in DVR's DTU format `dtu_dataset.zip`
    - SRN chair/car (128x128) `srn_*.zip`
      note the car set is a re-rendered version provided by Vincent Sitzmann

While we could have used a common data format, we chose to keep
DTU and ShapeNet (NMR) datasets in DVR's format and SRN data in the original SRN format.
Our own two-object data is in NeRF's format.
Data adapters are built into the code.

# Running the model (video generation)

Download all pretrained weight files from
<https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing>.
Extract this to `<project dir>/checkpoints/`, so that `<project dir>/checkpoints/dtu/pixel_nerf_latest` exists.


## ShapeNet 64x64

1. Download NMR ShapeNet renderings (see Datasets section, 1st link)
2. Download the pretrained shapenet 64 models
3. Run using 
    - `python gen_video.py  -n sn64 -c conf/default_mv.conf --gpu_id <GPU> --split test -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr -S 0 --ray_batch_size=20000`
    - For unseen category generalization:
      `python gen_video.py  -n sn64_unseen -c conf/default_mv.conf --gpu_id=<GPU> --split test -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr_gen -S 0 --ray_batch_size=20000`

Replace `<GPU>` with desired GPU id.  Replace `-S 0` with `-S <number>` to run on a different ShapeNet object id.
Replace `--split test` with `--split train | val` to use different split.
Adjust `--ray_batch_size` if running out of memory.

Pre-generated results for all ShapeNet objects with comparison may be found at <https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/>

## DTU

1. Download DTU dataset (see Datasets section). Extract to some directory, to get: `<data_root>/rs_dtu_4`
2. Download the pretrained DTU model
3. Run using `python gen_video.py  -n dtu -c conf/dtu.conf --gpu_id=<GPU> --split val -P '22 25 28'  -D <data_root>/rs_dtu_4 -F dvr_dtu -S 3  --ray_batch_size=20000 --black --scale 0.25`

Replace `<GPU>` with desired GPU id. Replace `-S 3` with `-S <number>` to run on a different scene.
Remove `--scale 0.25` to render at full reso (slow)

Note that for DTU, I only use train/val sets, where val is used for test. This is due to the very small size of the dataset. 
The model overfits to the train set significantly during training.

## Real Car Images

**Note: requires PointRend from detectron2.**
Install detectron2 by following https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.

1. Download any car image.
Place it in `<project dir>/input`. Some example images are shipped with the repo.
The car should be fully visible.
2. Download the pretrained *SRN car* model.
3. Run the preprocessor script: `python scripts/preproc.py`. This saves `input/*_normalize.png`.
If the result is not reasonable, PointRend didn't work; please try another imge.
4. Run `python eval_real.py`. Outputs will be in `<project dir>/output`

The Stanford Car dataset contains many example car images:
<https://ai.stanford.edu/~jkrause/cars/car_dataset.html>.
Note the normalization heuristic has been slightly modified compared to the paper. There may be some minor differences.
You can pass `-e -20' to `eval_real.py` to set the elevation higher in the generated video.

# Overview of flags

Generally, all scripts in the project take the following flags
- `-c <conf/*.conf>`: config file
- `-n <expname>`: experiment name, matching checkpoint directory name
- `-F <multi_obj | dvr | dvr_gen | dvr_dtu | srn>`: data format
- `-D <datadir>`: data directory. For SRN/multi_obj datasets with 
    separate directories e.g. `path/cars_train`, `path/cars_val`,
    put `-D path/cars`
- `--split <train | val | test>`: data set split
- `-S <subset_id>`: scene or object id to render
- `--gpu_id <GPU>`: GPU id to use

Please refer the the following table

| Name                       | expname -n | config -c            | data format -F | Data file                               | data dir -D         |
|----------------------------|------------|----------------------|----------------|-----------------------------------------|---------------------|
| ShapeNet category-agnostic | sn64       | conf/sn64.conf       | dvr            | NMR_Dataset.zip (from AWS)              | <path>/NMR_Dataset  |
| ShapeNet unseen category   | sn64_gen   | conf/sn64.conf       | dvr_gen        | NMR_Dataset.zip (from AWS) + genlist.py | <path> /NMR_Dataset |
| SRN chairs                 | srn_chair  | conf/default_mv.conf | srn            | srn_chairs.zip                          | <path>/chairs       |
| SRN cars                   | srn_car    | conf/default_mv.conf | srn            | srn_cars.zip                            | <path>/cars         |
| DTU                        | dtu        | conf/dtu.conf        | dvr_dtu        | dtu_dataset.zip                         | <path>/rs_dtu_4     |
| Two chairs                 | TBA        | TBA                  | multi_obj      | multi_chair_*.zip                       | <path>              |


# Quantitative evaluation instructions

The full, parallelized evaluation code is in `eval.py`. Note that this can be extremely slow.
Therefore we also provide `eval_approx.py` for *approximate* evaluation.

- Example `python eval_approx.py -F srn -D <srn_data>/cars -n srn_car`

# Training instructions

Check out `train/train.py`

# BibTeX

```
@misc{yu2020pixelnerf,
      title={pixelNeRF: Neural Radiance Fields from One or Few Images}, 
      author={Alex Yu and Vickie Ye and Matthew Tancik and Angjoo Kanazawa},
      year={2020},
      eprint={2012.02190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgements

Parts of the code were based on from kwea123's NeRF implementation: https://github.com/kwea123/nerf_pl.
Some functions are borrowed from DVR https://github.com/autonomousvision/differentiable_volumetric_rendering
and PIFu https://github.com/shunsukesaito/PIFu
