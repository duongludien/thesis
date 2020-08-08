# Real-time traffic-signs recognition using YOLOv3

Bachelor's Thesis

**Student**: Dương Lữ Điện

**Supervisor**: Trần Công Án, PhD.

We use YOLO algorithm in our dataset and try to detect some types of traffic signs in Vietnam: 102, 130, 131, 201a, 201b, 202, 203, 205, 207, 208, 209, 221, 224, 225, 233, 245, 302, 303, 423, crowded, end_crowded, traffic_light

## Important Notice

This project was published under GNU General Public License v3.0. Make sure that you have read the [LICENSE](LICENSE) carefully before using it in your project.

## Resources

- [Report PDF](https://drive.google.com/file/d/1wOnbA15dvSQY1XteTBcIkFYyC2L7IqAO/view?usp=sharing)
- [Fozen model](https://drive.google.com/file/d/16rkONK0Vm-mB3Ra6U9w2L6R36hyU13wT/view?usp=sharing)
- [Slides](https://drive.google.com/file/d/15ncYhdvnVJTkGg4Mnxi1znGfXdRHvqVd/view?usp=sharing)
- Dataset: send an email to duongludien@gm*il.com
- TensorFlow checkpoint: send an email to duongludien@gm*il.com

## How to use?

1. Navigate to your local repo.
```
$ cd YOUR_LOCAL_REPO/yolov3/
```

2. Create a virtual environment and install requirements
```
$ virtualenv test_env --python=python3.6
$ source test_env/bin/activate
(test_env)$ pip install -r requirements.txt
```
3. While requirements are being installed, download frozen model in the **Resources** section and put it into `yolov3` directory.

4. After all finished, run the demonstration
- For predicting images:
```
(test_env)$ python predict_frozen.py YOUR_IMAGES_DIRECTORY_PATH
```
Press **Left** for previous, **q** for exiting and another key for next

- For predicting videos:
```
(test_env)$ python predict_video_frozen.py YOUR_VIDEOS_DIRECTORY_PATH
```
Press **q** for exiting

**Note**: If you want to develop your own model from my code, install all requirements yourself :)
