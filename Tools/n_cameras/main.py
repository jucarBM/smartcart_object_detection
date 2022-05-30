# main of the n_cameras tool
from mul_camera import Camera, compare2cameras
from threading import Thread
# init function of python script

if __name__ == "__main__":
    # create camera objects
    cam1 = Camera(0, "zero")
    cam2 = Camera(1, "one")
    main_thread = Thread(target=compare2cameras, args=(cam1, cam2,))
    # run the cameras
    cam1.start()
    cam2.start()
    main_thread.start()