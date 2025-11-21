# multiperson-personalized
This repository contains the code to evaluate the work presented in the article [Extending 3D body pose estimation for robotic-assistive therapies of autistic children](https://ieeexplore.ieee.org/document/10719777). 

<pre lang="markdown"> @INPROCEEDINGS{Santos2024,
  author={Santos, Laura and Carvalho, Bernardo and Barata, Catarina and Santos-Victor, José},
  booktitle={2024 10th IEEE RAS/EMBS International Conference for Biomedical Robotics and Biomechatronics (BioRob)}, 
  title={Extending 3D Body Pose Estimation for Robotic-Assistive Therapies of Autistic Children}, 
  year={2024},
  volume={},
  number={},
  pages={520-525},
  keywords={Solid modeling;Adaptation models;Pediatrics;Three-dimensional displays;Computational modeling;Robot vision systems;Pose estimation;Linear regression;Medical treatment;Cameras},
  doi={10.1109/BioRob60516.2024.10719777}} </pre>


## Dependencies & Installation
* Forked-Multiperson https://github.com/laurajsantos/multiperson-cuda11.8.git
* BEV https://github.com/Arthur151/ROMP

## Focal length model creation
To create the model of the focal length, discover the focal length of at least two people staying in the front of a camera at a known distance and use the code

<pre lang="markdown"> bash run_multiperson_focal.sh </pre>

After obtaining the values of focal length use them, the height of the people and the distance they were in front of the camera to construct the linear model in <pre lang="markdown"> focal_length.m </pre> 

For any new person, subtitute the values a,b in <pre lang="markdown"> modelo_focal_length.m </pre> With the ones found in <pre lang="markdown"> focal_length.m </pre>


## Body pose estimation
Use the new values of focal length f1 and f2 in <pre lang="markdown"> run_multiperson_2.sh </pre> by doing:

<pre lang="markdown"> bash run_multiperson_2.sh f1 f2 </pre>







