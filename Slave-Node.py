import socket
import numpy as np
import threading
import keyboard  # Requires the 'keyboard' library
import time

class SlaveNode:
    def __init__(self, broadcast_port=5001):
        self.master_ip = None
        self.master_port = None
        self.sock = None
        self.broadcast_port = broadcast_port
        self.running = True
        self.connect_to_master()

    def connect_to_master(self):
        while self.running:
            try:
                self.listen_for_master()
                self.listen_for_tasks()
            except (ConnectionResetError, ConnectionRefusedError, socket.error) as e:
                print(f"Connection error: {e}. Reconnecting...")
                time.sleep(2)
                continue

    def listen_for_master(self):
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        broadcast_socket.bind(("", self.broadcast_port))

        print("Listening for broadcast from Master...")
        while self.running:
            data, addr = broadcast_socket.recvfrom(1024)
            message = data.decode()
            self.master_ip, self.master_port = message.split(':')
            self.master_port = int(self.master_port)
            print(f"Received broadcast from Master: {self.master_ip}:{self.master_port}")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.master_ip, self.master_port))
            print(f"Connected to Master at {self.master_ip}:{self.master_port}")
            break

    def listen_for_exit(self):
        def check_for_exit():
            while True:
                if keyboard.is_pressed('F8') or keyboard.is_pressed('ctrl+x'):
                    print("Slave Node exiting...")
                    self.running = False
                    if self.sock:
                        self.sock.close()
                    exit(0)
        threading.Thread(target=check_for_exit, daemon=True).start()

    def listen_for_tasks(self):
        self.listen_for_exit()
        try:
            while self.running:
                print("Waiting for work packets...")
                data = self.sock.recv(1024)  # Receive the matrix dimensions first
                if not data:
                    print("Disconnected from Master.")
                    break
                print(f"Data packet received: {data.decode()}")
                sub_matrix_a_rows, sub_matrix_a_cols, matrix_b_rows, matrix_b_cols = map(int, data.decode().split(','))
                self.sock.sendall(b'1')  # Acknowledgement to Master

                print("Receiving matrix data...")
                sub_matrix_a_size = sub_matrix_a_rows * sub_matrix_a_cols
                matrix_b_size = matrix_b_rows * matrix_b_cols

                sub_matrix_a_data = self.recv_exact(8 * sub_matrix_a_size)
                print(f"Matrix A received: {sub_matrix_a_size * 8} bytes")

                matrix_b_data = self.recv_exact(8 * matrix_b_size)
                print(f"Matrix B received: {matrix_b_size * 8} bytes")

                print("Starting matrix multiplication...")
                start_time = time.time()

                sub_matrix_a = np.frombuffer(sub_matrix_a_data, dtype=np.float64).reshape((sub_matrix_a_rows, sub_matrix_a_cols))
                matrix_b = np.frombuffer(matrix_b_data, dtype=np.float64).reshape((matrix_b_rows, matrix_b_cols))
                result = np.dot(sub_matrix_a, matrix_b)

                end_time = time.time()
                computation_time = end_time - start_time

                print(f"Matrix multiplication done in {computation_time:.4f} seconds. Returning results...")

                if self.send_results(result, computation_time):
                    print("Results and time returned successfully.")
                else:
                    print("Failed to return results and time. Retrying...")
                    if self.send_results(result, computation_time):
                        print("Retry successful. Results and time returned.")
                    else:
                        print("Retry failed. Aborting operation.")

                print("*" * 40)

        except (ConnectionResetError, ConnectionRefusedError, socket.error) as e:
            print(f"Connection error: {e}. Reconnecting...")

    def recv_exact(self, size):
        buffer = bytearray()
        while len(buffer) < size:
            packet = self.sock.recv(size - len(buffer))
            if not packet:
                break
            buffer.extend(packet)
        return buffer

    def send_results(self, result, computation_time):
        try:
            self.sock.sendall(result.tobytes())  # Send the result matrix
            self.sock.sendall(np.array([computation_time], dtype=np.float64).tobytes())  # Send the time taken as a float64
            return True
        except Exception as e:
            print(f"Error sending results: {e}")
            return False

if __name__ == "__main__":
    slave = SlaveNode()
    slave.connect_to_master()
