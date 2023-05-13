import socket
import tqdm
import os
import CGL_test_server as cgl_server
from multiprocessing import Process, Queue
import time

# device's IP address
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

# create the server socket
# TCP socket
s = socket.socket()

# bind the socket to our local address
s.bind((SERVER_HOST, SERVER_PORT))

CGL_q = Queue()
a = Process(target=cgl_server.initialize, args=(CGL_q,))
a.start()

while True:
    s.listen(5)
    print("[*] Listening as {}:{}".format(SERVER_HOST, SERVER_PORT))

    client_socket, address = s.accept() 
    print("[+] {} is connected.".format(address))
    
    received = client_socket.recv(BUFFER_SIZE).decode()
        
    filename, filesize, folder_name = received.split(SEPARATOR)
    filename = os.path.basename(filename)
    filesize = int(filesize)

    if not os.path.isdir("server/" + folder_name):
        os.mkdir("server/" + folder_name)

    progress = tqdm.tqdm(range(filesize), "Receiving {}".format(filename), unit="B", unit_scale=True, unit_divisor=1024)
    
    with open("server/" + folder_name + "/" + filename, "wb") as f:
        while True:
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                break
            f.write(bytes_read)
            progress.update(len(bytes_read))
    
    client_socket.close()
    CGL_q.put(folder_name + "/" + filename)
    
    while not os.path.exists("server_output/" + folder_name + "/" + filename):
        pass
    
    client_socket, address = s.accept() 
    print("[+] {} is connected.".format(address))


    filename = "server_output/" + folder_name + "/" + filename
    
    filesize = os.path.getsize(filename)
    
    client_socket.send("{}{}{}".format(filename, SEPARATOR, filesize).encode())
    time.sleep(1)
    progress = tqdm.tqdm(range(filesize), "Sending {}".format(filename), unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            print("sending output")
            if not bytes_read:
                # file transmitting is done
                break
            client_socket.sendall(bytes_read)
            progress.update(len(bytes_read))        

    client_socket.close()

    time.sleep(1)

s.close()
