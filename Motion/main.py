import object_detect
from Tools.n_cameras import mul_camera


# main function
def main():
    cam1 = mul_camera.Camera(0, "zero",
                             function_preprocess=object_detect.get_background)
    cam1.start()


if __name__ == "__main__":
    main()
