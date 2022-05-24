# main of the n_cameras tool
from mul_camera import Camera
# init function of python script

if __name__ == "__main__":
    # create camera objects
    cam1 = Camera(0, "zero", "other")
    cam2 = Camera(1, "one", "other")
    # run the cameras
    cam1.start()
    cam2.start()
