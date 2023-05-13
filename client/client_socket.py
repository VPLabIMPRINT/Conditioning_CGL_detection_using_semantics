import socket
import tqdm
import os
import time

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step

# the ip address or hostname of the server, the receiver
host = "0.0.0.0"
# the port, let's use 5001
port = 5001

for filename in os.listdir("images"):
    # create the client socket
    s = socket.socket()

    print(f"[+] Connecting to {host}:{port}")
    s.connect((host, port))
    print("[+] Connected.")

    filename = "images/" + filename 
    # get the file size
    filesize = os.path.getsize(filename)

    # send the filename and filesize
    s.send(f"{filename}{SEPARATOR}{filesize}".encode())

    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
    
    s.close()

    # create the client socket
    s = socket.socket()

    print(f"[+] Connecting to {host}:{port}")
    s.connect((host, port))
    print("[+] Connected.")

    received = s.recv(BUFFER_SIZE).decode()
    
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)

    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open("output/" + filename, "wb") as f:
        while True:
            # print("dfsds")
            # read 1024 bytes from the socket (receive)
            bytes_read = s.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                # file transmitting is done
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    
    #close the socket
    s.close()

    time.sleep(1)

    