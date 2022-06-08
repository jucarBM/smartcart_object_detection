#!/usr/bin/python3
import time
import serial

print("UART Demonstration Program")
print("NVIDIA Jetson Nano Developer Kit")

def serial ():
	serial_port = serial.Serial(
		port="/dev/ttyTHS1",
		baudrate=115200,
		bytesize=serial.EIGHTBITS,
		parity=serial.PARITY_NONE,
		stopbits=serial.STOPBITS_ONE
	)
# Wait a second to let the port initialize
time.sleep(1)
sku = 7750243057448

def mandar_sku(mensaje):
	
	if serial_port.inWaiting() > 0:
		data=serial_port.readline(5)
		data=data.decode()
		data_str = str(data)
		if data_str == "Hola2":
			serial_port.write(str(mensaje).encode())

def HandShake(mensajee):
	serial_port.write(mensajee.encode())
		
while True:
	'''
	original ="Hola"

		
	mensajitos(original)
	
	if serial_port.inWaiting() > 0:
		data=serial_port.readline(5)
		data=data.decode()
		data_str = str(data)
		if data_str == "Hola2":
			
			mensajitos(sku)
			print("Hola2")
		else:
			original ="Hola"
			
			mensajitos(original)
			
	
	
	
	'''
	
	
	HandShake("Hola")
	
	#if serial_port.inWaiting() > 0:
	data=serial_port.read(5)
	data=data.decode()
	data_str = str(data)
	
	if data_str == "Hola2":
		serial_port.write('7750243057448'.encode())
		print(data_str)
		
	else:
		HandShake("Hola")
		
				
			
			
			
	


	
	#time.sleep(1)
	
	
	#data =serial.read()
	
	
	

	
#	if data == "Hola2":
#	    serial_port.write(sku)  #envio sku
	    # serial_port.write("1252632".encode())
	    # serial_port.write("7750243057448".encode())
	    # serial_port.write("hola estas recibiendo".encode())
#	    time.sleep(1)
#	elif data == "Falso":
#	    serial_port.write(sku)
#
#	elif data == "Correcto":
#	    serial_port.write(sku)

#	else:
#	    serial_port.write("Hola") 

	        # else:



    # serial_port.write("657894621352216482448".encode())
    # # time.sleep(1)
    # serial_port.write("1252632".encode())
    # # time.sleep(1)
    # serial_port.write("7750243057448".encode())
    # # time.sleep(1)
    # serial_port.write("hola estas recibiendo".encode())
    # # while True:
        # if serial_port.inWaiting() > 0:
        #     data = serial_port.read()
        #     print(data)
        #     serial_port.write(data)
        #     # if we get a carriage return, add a line feed too
        #     # \r is a carriage return; \n is a line feed
        #     # This is to help the tty program on the other end 
        #     # Windows is \r\n for carriage return, line feed
        #     # Macintosh and Linux use \n
        #     if data == "\r".encode():
        #         # For Windows boxen on the other end
        #         serial_port.write("\n".encode())



