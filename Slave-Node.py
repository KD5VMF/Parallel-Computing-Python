import socket
import numpy as np

class SlaveNode:
    def __init__(self, master_ip, master_port):
        self.master_ip = master_ip
        self.master_port = master_port
        self.connect_to_master()

    def connect_to_master(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.master_ip, self.master_port))
        print(f"Connected to Master at {self.master_ip}:{self.master_port}")
        self.listen_for_tasks()

    def listen_for_tasks(self):
        while True:
            data = self.sock.recv(4096)
            if data:
                matrix_a_size = int(len(data) / 8)
                matrix_a = np.frombuffer(data[:matrix_a_size], dtype=np.float64).reshape((-1, matrix_a_size))
                matrix_b = np.frombuffer(data[matrix_a_size:], dtype=np.float64).reshape((matrix_a_size, -1))

                result = np.dot(matrix_a, matrix_b)
                self.sock.sendall(result.tobytes())

if __name__ == "__main__":
    master_ip = input("Enter Master IP: ")
    master_port = int(input("Enter Master Port: "))
    slave = SlaveNode(master_ip, master_port)
