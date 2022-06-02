import time
import serial
print("UART Demonstration Program")
print("NVIDIA Jetson Nano Developer Kit")


serial_port = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE
)
# Wait a second to let the port initialize
time.sleep(1)


def HandShake(mensajee):
	serial_port.write(str(mensajee).encode())
