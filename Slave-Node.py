import socket
import numpy as np
import threading
import keyboard  # Requires the 'keyboard' library

class SlaveNode:
    def __init__(self, broadcast_port=5001):
        self.master_ip = None
        self.master_port = None
        self.sock = None
        self.broadcast_port = broadcast_port
        self.listen_for_master()

    def listen_for_master(self):
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        broadcast_socket.bind(("", self.broadcast_port))

        print("Listening for broadcast from Master...")
        while True:
            data, addr = broadcast_socket.recvfrom(1024)
            message = data.decode()
            self.master_ip, self.master_port = message.split(':')
            self.master_port = int(self.master_port)
            print(f"Received broadcast from Master: {self.master_ip}:{self.master_port}")
            self.connect_to_master()
            break  # Stop listening once connected to a Master

    def connect_to_master(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.master_ip, self.master_port))
        print(f"Connected to Master at {self.master_ip}:{self.master_port}")
        self.listen_for_tasks()

    def listen_for_exit(self):
        def check_for_exit():
            while True:
                if keyboard.is_pressed('F8') or keyboard.is_pressed('ctrl+x'):
                    print("Slave Node exiting...")
                    self.sock.close()
                    exit(0)
        threading.Thread(target=check_for_exit, daemon=True).start()

    def listen_for_tasks(self):
        try:
            while True:
                print("Waiting for work packets...")
                data = self.sock.recv(1024)  # Receive the matrix dimensions first
                if data:
                    print(f"Data packet received: {data.decode()}")
                    sub_matrix_a_rows, sub_matrix_a_cols, matrix_b_rows, matrix_b_cols = map(int, data.decode().split(','))
                    self.sock.sendall(b'1')  # Acknowledgement to Master

                    print("Receiving matrix data...")
                    # Receive the actual matrix data
                    sub_matrix_a_size = sub_matrix_a_rows * sub_matrix_a_cols
                    matrix_b_size = matrix_b_rows * matrix_b_cols

                    sub_matrix_a_data = self.recv_exact(8 * sub_matrix_a_size)
                    print(f"Matrix A received: {sub_matrix_a_size * 8} bytes")

                    matrix_b_data = self.recv_exact(8 * matrix_b_size)
                    print(f"Matrix B received: {matrix_b_size * 8} bytes")

                    print("Starting matrix multiplication...")
                    sub_matrix_a = np.frombuffer(sub_matrix_a_data, dtype=np.float64).reshape((sub_matrix_a_rows, sub_matrix_a_cols))
                    matrix_b = np.frombuffer(matrix_b_data, dtype=np.float64).reshape((matrix_b_rows, matrix_b_cols))

                    # Perform matrix multiplication
                    result = np.dot(sub_matrix_a, matrix_b)
                    print("Matrix multiplication done. Returning results...")

                    if self.send_results(result):
                        print("Results returned successfully.")
                    else:
                        print("Failed to return results. Retrying...")
                        if self.send_results(result):
                            print("Retry successful. Results returned.")
                        else:
                            print("Retry failed. Aborting operation.")

                    print("*" * 40)

        except Exception as e:
            print(f"Error occurred: {e}")
            self.sock.close()

    def recv_exact(self, size):
        """Ensure that we receive exactly `size` bytes from the connection."""
        buffer = bytearray()
        while len(buffer) < size:
            packet = self.sock.recv(size - len(buffer))
            if not packet:
                break
            buffer.extend(packet)
        return buffer

    def send_results(self, result):
        """Send the result back to the Master, with a simple retry mechanism."""
        try:
            self.sock.sendall(result.tobytes())
            return True
        except Exception as e:
            print(f"Error sending results: {e}")
            return False

if __name__ == "__main__":
    slave = SlaveNode()
