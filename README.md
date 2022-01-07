# Knowing Human Weight from A Single Image

 **Body weight as one of the biometric traits has been
studied in both the forensic and medical domains. However,
estimating weight directly from 2D images is particularly
challenging, since the visual inspection is rather sensitive to the
distance between the subject and camera, even for the frontal
view images. In this case, the widely used Body Mass Index
(BMI) which is associated with body height and weight can be
employed as a measure of weight to indicate the health conditions.
Previous works on the estimation of BMI have predominantly
focused on using multiple 2D images, 3D images, or facial images,
however, these cues are not always available. To address this issue,
we explore the feasibility of obtaining BMI from a single 2D
body image with a dual-branch regression framework proposed
in this work. More specifically, the framework comprises an
anthropometric feature computation branch and a deep learning-based feature extraction branch. One aggregation layer maps all
the features to an estimated BMI value. In addition, a new public
2Dimage-to-BMI dataset is collected and released to facilitate
the study, which contains 4189 images (1477 males and 2712
females) from around 3000 subjects with the attributes including
gender, age, height, and weight. Extensive experiments confirm
that the proposed framework combining anthropometric features
and deep features outperforms the single-type feature approaches
in most cases on BMI estimation.**

<div align=center>
<img src="https://github.com/FVL2020/2DImage2BMI/blob/main/framework/framework.jpg">
</div>

## Install

Our code is tested with PyTorch 1.4.0, CUDA 10.0 and Python 3.6. It may work with other versions.

You will need to install some python dependencies.

You will need to install some python dependencies(either `conda install` or `pip install`)

```
scikit-learn
scipy
tensorboardX
opencv-python
pandas
```

We use the pretrained model in [detectron2](https://github.com/facebookresearch/detectron2), so you need to install the project following their installation instructions.

The Pos2Seg model, Human Parse model and deep feature extracted model are stored in [google drive](https://drive.google.com/file/d/1BsIbUWktXxIe75fM_JWphvYB0yjV-RZy/view?usp=sharing), you can downlown them and put them in current directory.

## Testing

You can easily get the test result by running
```
python Regression.py
``` 

## Result
<div align=center>
<img src="https://github.com/FVL2020/2DImage2BMI/blob/main/framework/result.jpg">
</div>

