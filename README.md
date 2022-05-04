# Visual-Inertial-Odometry-MSCKF-Stereo

## MSCKF

### Requirements
Python 3.6+ <br>
numpy<br>
scipy<br>
cv2<br>
numba<br>
pangolin (optional, for trajectory/poses visualization)

### To Run
python vio.py --path path/to/your/Dataset

### Results

![alt text](https://github.com/NonStopEagle137/Visual-Inertial-Odometry-MSCKF-Stereo/blob/master/Screenshot%202022-05-03%20203319.png)

## ESKF

### Requirements
Python 3.6+ <br>
pyyaml<br>
cvxopt<br>
matplotlib == 3.2.2 <br>
numpy <br> 
scipy <br>
timeout_decorator <br>

### To Run 
Run the file eskf_vio, please change the dataset path variable in the file to where your dataset.

### Results

![alt text](https://github.com/NonStopEagle137/Visual-Inertial-Odometry-MSCKF-Stereo/blob/master/Screenshot%202022-05-03%20203244.png)
