# SplineGen: data-driven, generative AI-based B-spline modeling

- By Lizhen Zhu and Qiang Zou
- Email: qzou.code@gmail.com
- Webpage: https://qiang-zou.github.io/
- Latest Release: 2024.10.21

## !important
The source code was developed as a byproduct of the projects and methods presented in [1,2]. It promotes a new CAD modeling paradigm---data-driven, generative B-spline modeling---focusing on parameterization and knot placement for approximating unorganized points. The core principle is to replace manual heuristics with data-driven insights, enhancing modeling accuracy, efficiency, and robustness.

It can be run with Pytorch 2.0.1 + CUDA 11.8 on the operating system Ubuntu 22.04 LTS. Windows, Mac.


1.Copyright
-----------

- SplineGen is GNU licensed. It is developed and maintained by Lizhe Zhu and Qiang Zou for research use. All rights about the program are reserved by Qiang Zou. This Python source code is available only to a primary user for academic purposes. No secondary use, such as copy, distribution, diversion, business purpose, etc., is allowed. In no event shall the author be liable to any party for direct, indirect, special, incidental, or consequential damage arising out of the use of this program. SplineGen is self-contained.


2.Download
----------

- The source code can be downloaded from: [https://github.com/Qiang-Zou/SplineGen](https://github.com/Qiang-Zou/SplineGen)
  

3.Installing (Windows/Linux/Mac + Pytorch 2.0.1 + CUDA 11.8)
-------------------------------------------

- Environment setup

    ```shell
    conda env create -f environment.yml
    conda activate splinegen
    # install geomdl with pip
    pip install geomdl
    ```

- To create the dataset, a curve intersection detect module is also needed installed.

    ```shell
    cd ops/3d_intersect_detect
    pip install .
    ```

4.Usage
-------

- You can take four steps to train the SplineGen with src/splinegen/main.py

    ```shell
    # train encoder
    python src/splinegen/main.py train_encoder --dataset_path /path/to/dataset 
    # train knot decoder
    python src/splinegen/main.py train_knot_decoder --dataset_path /path/to/dataset --encoder_path /path/to/encoder/path
    # train parameter decoder
    python src/splinegen/main.py train_param_decoder --dataset_path /path/to/dataset --knot_path /path/to/knot/generation/model
    # train knot decoder
    python src/splinegen/main.py train_diff_approximation --dataset_path /path/to/dataset --base_model_path /path/to/knot&param/generation/model 
    ```

- You can test the model by the command following

    ```shell
    python src/splinegen/main.py test --dataset_path /path/to/dataset --model_path /path/to/splinegen 
    ```

- To create 2D/3D dataset, run

    ```shell
    python src/splinegen/process/gen2d_curve.py
    python src/splinegen/process/gen3d_curve.py
    ```

5.References
-------------

- [1] Qiang Zou, Lizhen Zhu, Jiayu Wu, and Zhijie Yang, "SplineGen: Approximating unorganized points through generative AI." Computer-Aided Design, vol.178, 103809, 2025. https://doi.org/10.1016/j.cad.2024.103809
- [2] Qiang Zou , Yincai Wu , Zhenyu Liu , Weiwei Xu , Shuming Gao. "Intelligent CAD 2.0." Visual Informatics (2024). https://doi.org/10.1016/j.visinf.2024.10.001
