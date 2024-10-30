# SwinCS-VFIT

Video Frame Interpolation Transformer based on Shifted-window and Cross-Scale window.

This repo is the implementation of SwinCS-VFIT. [Paper](https://www.arocmag.com/abs/2023.07.0344)

## Packages

The following pakages are required to run the code:

* python==3.7.6
* pytorch==1.5.1
* cudatoolkit==10.1
* torchvision==0.6.1
* cupy==7.5.0
* pillow==8.2.0
* einops==0.3.0

## Train

* Download the [Vimeo-90K septuplets](http://toflow.csail.mit.edu/) dataset.

* Then train SwinCS-VFIT-B using default training configurations:

```sh
python main.py --model SwinCS_VFIT_B --dataset vimeo90K_septuplet --data_root <dataset_path>
```

Training SwinCS-VFIT-S is similiar to above, just change ```model``` to SwinCS_VFIT_S.

## Test

After training, you can evaluate the model with following command:

```sh
python test.py --model SwinCS_VFIT_B --dataset vimeo90K_septuplet --data_root <dataset_path> --load_from checkpoints/SwinCS_VFIT_B/model_best.pth
```

You can also evaluate SwinCS-VFIT using our weight [here](https://drive.google.com/drive/folders/1J8Onvc3OjuGnkCzJ88FQ5oIlObxRxP4Q?usp=sharing).

More datasets for evaluation:

* [UCF](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fdbihqk5deobn0f7%2Fucf101_extracted.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNE8CyLdENKhJf2eyFUWu6G2D1iJUQ)
* [Davis](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2F9t6x7fi9ui0x6bt%2Fdavis-90.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNG7jT-Up65GD33d1tUftjPYNdQxkg)

## Interpolate

Folders structure like this:

```sh
.
├── inter_data
│   ├── folder1
│   │   ├── im1.jpg
│   │   ├── ...
│   │   └── im7.jpg
│   └── ...
├── out_data
└── SwinCS-VFIT
    ├── interpolate_demo1.py
    └── interpolate_demo2.py
```

Run interpolation with SwinCS-VFIT-B:

```sh
python interpolate_demo1.py --model SwinCS_VFIT_B --load_from checkpoints/SwinCS_VFIT_B/model_best.pth
```

Or specify your own input and output dir:

```sh
python interpolate_demo1.py --model SwinCS_VFIT_B --load_from checkpoints/SwinCS_VFIT_B/model_best.pth --img_path <path/to/inter_data> --out_path <path/to/out_data>
```

Interpolating frame with SwinCS-VFIT-S is similiar.

## References

Some other great video interpolation resources that we benefit from:

* VFIT: Video Frame Interpolation Transformer, CVPR 2022 [Code](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer)
* VFIformer: Video Frame Interpolation with Transformer, CVPR 2022 [Code](https://github.com/dvlab-research/VFIformer)
* FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation, arXiv 2021 [Code](https://github.com/tarun005/FLAVR)
* QVI: Quadratic Video Interpolation, NeurIPS 2019 [Code](https://github.com/xuxy09/QVI)
* AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation, CVPR 2020 [Code](https://github.com/HyeongminLEE/AdaCoF-pytorch)

Thanks a lot!
