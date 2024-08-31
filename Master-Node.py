import socket
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np

class MasterNode:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.slave_connections = []
        self.window = tk.Tk()
        self.window.title("Master Node")

        self.info_label = tk.Label(self.window, text=f"IP: {self.get_ip_address()} | Port: {self.port}")
        self.info_label.pack()

        self.connection_listbox = tk.Listbox(self.window)
        self.connection_listbox.pack()

        self.matrix_size_var = tk.StringVar(value="1024x1024")
        self.matrix_entry = tk.Entry(self.window, textvariable=self.matrix_size_var)
        self.matrix_entry.pack()

        self.ready_button = tk.Button(self.window, text="Ready", command=self.ready)
        self.ready_button.pack()

        self.go_button = tk.Button(self.window, text="Go", command=self.start_computation)
        self.go_button.pack()

        threading.Thread(target=self.start_server).start()

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Master Node is listening on {self.host}:{self.port}")

        while True:
            client_socket, address = server_socket.accept()
            self.slave_connections.append((client_socket, address))
            self.connection_listbox.insert(tk.END, f"Connected: {address}")
            print(f"New Slave connected from {address}")

    def ready(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
        else:
            messagebox.showinfo("Info", "Ready to start the computation!")

    def start_computation(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
            return

        matrix_size = self.matrix_size_var.get()
        rows, cols = map(int, matrix_size.split("x"))
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)

        sub_matrix_size = rows // len(self.slave_connections)
        start_row = 0

        for conn, address in self.slave_connections:
            sub_matrix_a = matrix_a[start_row:start_row + sub_matrix_size, :]
            conn.sendall(sub_matrix_a.tobytes())
            conn.sendall(matrix_b.tobytes())
            start_row += sub_matrix_size

        results = []
        for conn, _ in self.slave_connections:
            data = conn.recv(4096)
            result_sub_matrix = np.frombuffer(data, dtype=np.float64).reshape(sub_matrix_size, rows)
            results.append(result_sub_matrix)

        final_result = np.vstack(results)
        messagebox.showinfo("Info", f"Matrix multiplication completed.\nResult: {final_result}")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    master = MasterNode()
    master.run()
