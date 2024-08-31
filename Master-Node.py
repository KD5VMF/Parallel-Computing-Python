import socket
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import time

class MasterNode:
    def __init__(self, host="0.0.0.0", port=5000, broadcast_port=5001):
        self.host = host
        self.port = port
        self.broadcast_port = broadcast_port
        self.slave_connections = []
        self.slave_performance = []
        self.broadcasting = True
        self.window = tk.Tk()
        self.window.title("Master Node")
        self.window.geometry("400x300")
        self.window.resizable(False, False)

        # Create main frame
        main_frame = tk.Frame(self.window, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Server info section
        server_frame = tk.LabelFrame(main_frame, text="Server Info", padx=10, pady=10)
        server_frame.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(server_frame, text=f"IP: {self.get_ip_address()} | Port: {self.port}")
        self.info_label.pack(anchor="w")

        # Connected slaves section
        connection_frame = tk.LabelFrame(main_frame, text="Connected Slaves", padx=10, pady=10)
        connection_frame.pack(fill=tk.BOTH, expand=True)

        self.connection_listbox = tk.Listbox(connection_frame, height=5)
        self.connection_listbox.pack(fill=tk.BOTH, expand=True)

        # Matrix size and control buttons
        control_frame = tk.Frame(main_frame, padx=10, pady=10)
        control_frame.pack(fill=tk.BOTH, expand=True)

        matrix_size_label = tk.Label(control_frame, text="Matrix Size (NxN):")
        matrix_size_label.grid(row=0, column=0, sticky="w")

        self.matrix_size_var = tk.StringVar(value="1024x1024")
        self.matrix_entry = tk.Entry(control_frame, textvariable=self.matrix_size_var, width=10)
        self.matrix_entry.grid(row=0, column=1, sticky="e")

        self.ready_button = tk.Button(control_frame, text="Ready", command=self.ready)
        self.ready_button.grid(row=1, column=0, pady=5)

        self.go_button = tk.Button(control_frame, text="Go", command=self.start_computation)
        self.go_button.grid(row=1, column=1, pady=5)
        self.go_button.config(state=tk.DISABLED)

        threading.Thread(target=self.start_server).start()
        threading.Thread(target=self.broadcast_hello).start()  # Start broadcasting "hello" messages

        # Bind keys for exit
        self.window.bind('<F8>', self.exit_program)
        self.window.bind('<Control-x>', self.exit_program)

    def exit_program(self, event=None):
        self.broadcasting = False
        self.window.destroy()
        for conn, _ in self.slave_connections:
            conn.close()
        print("Master Node exited.")
        exit(0)

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
            self.slave_connections.append(client_socket)  # Only store the client_socket, not the tuple
            self.connection_listbox.insert(tk.END, f"Connected: {address}")
            print(f"New Slave connected from {address}")

    def broadcast_hello(self):
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        message = f"{self.get_ip_address()}:{self.port}"
        while self.broadcasting:
            broadcast_socket.sendto(message.encode(), ('<broadcast>', self.broadcast_port))
            print(f"Broadcasting hello: {message}")
            threading.Event().wait(2)  # Broadcast every 2 seconds

    def ready(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
        else:
            self.broadcasting = False  # Stop broadcasting
            self.ready_button.config(state=tk.DISABLED)
            self.go_button.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Ready to start the computation!")

    def start_computation(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
            return

        matrix_size = self.matrix_size_var.get()
        rows, cols = map(int, matrix_size.split("x"))
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)

        num_slaves = len(self.slave_connections)
        sub_matrix_size = rows // num_slaves
        extra_rows = rows % num_slaves  # In case rows are not perfectly divisible by number of slaves
        start_row = 0

        self.slave_performance.clear()  # Clear previous performance data

        try:
            for i, conn in enumerate(self.slave_connections):  # Iterate directly over the socket connections
                rows_to_send = sub_matrix_size + (1 if i < extra_rows else 0)
                sub_matrix_a = matrix_a[start_row:start_row + rows_to_send, :]
                sub_matrix_shape = sub_matrix_a.shape
                matrix_b_shape = matrix_b.shape

                # Send matrix shapes first
                conn.sendall(f"{sub_matrix_shape[0]},{sub_matrix_shape[1]},{matrix_b_shape[0]},{matrix_b_shape[1]}".encode())
                conn.recv(1)  # Acknowledgement from Slave

                start_time = time.time()

                # Send matrix data
                conn.sendall(sub_matrix_a.tobytes())
                conn.sendall(matrix_b.tobytes())
                start_row += rows_to_send

                # Record the time before receiving the results
                self.slave_performance.append({
                    'address': conn.getpeername(),
                    'rows': rows_to_send,
                    'start_time': start_time,
                    'end_time': None
                })

            results = []
            for i, (conn, performance) in enumerate(zip(self.slave_connections, self.slave_performance)):
                rows_to_receive = performance['rows']
                data = self.recv_exact(conn, 8 * rows_to_receive * cols)
                result_sub_matrix = np.frombuffer(data, dtype=np.float64).reshape(rows_to_receive, cols)
                results.append(result_sub_matrix)

                # Record end time and compute duration
                performance['end_time'] = time.time()
                performance['duration'] = performance['end_time'] - performance['start_time']

            final_result = np.vstack(results)
            self.show_results(final_result)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during computation: {e}")

    def recv_exact(self, conn, size):
        """Ensure that we receive exactly `size` bytes from the connection."""
        buffer = bytearray()
        while len(buffer) < size:
            packet = conn.recv(size - len(buffer))
            if not packet:
                break
            buffer.extend(packet)
        return buffer

    def show_results(self, final_result):
        result_window = tk.Toplevel(self.window)
        result_window.title("Computation Results")
        result_window.geometry("700x500")

        result_frame = tk.Frame(result_window)
        result_frame.pack(fill=tk.BOTH, expand=True)

        result_label = tk.Label(result_frame, text="Matrix Multiplication Results", font=("Arial", 14))
        result_label.pack(pady=10)

        # Table for displaying each slave's performance
        columns = ("Address", "Rows Processed", "Time Taken (s)")
        tree = ttk.Treeview(result_frame, columns=columns, show="headings")
        tree.heading("Address", text="PC Address")
        tree.heading("Rows Processed", text="Rows Processed")
        tree.heading("Time Taken (s)", text="Time Taken (s)")

        for performance in self.slave_performance:
            tree.insert("", "end", values=(
                performance['address'],
                performance['rows'],
                f"{performance['duration']:.4f}"
            ))

        tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        button_frame = tk.Frame(result_window)
        button_frame.pack(fill=tk.X, pady=10)

        save_button = tk.Button(button_frame, text="Save Results", command=lambda: self.save_results(final_result))
        save_button.pack(side=tk.LEFT, padx=10)

        close_button = tk.Button(button_frame, text="Close", command=lambda: self.close_results_window(result_window))
        close_button.pack(side=tk.RIGHT, padx=10)

    def save_results(self, final_result):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(np.array2string(final_result, separator=', '))
            messagebox.showinfo("Saved", f"Results saved to {file_path}")

    def close_results_window(self, window):
        window.destroy()
        self.reset_for_new_task()

    def reset_for_new_task(self):
        messagebox.showinfo("Ready", "The system is ready for a new task.")
        self.connection_listbox.delete(0, tk.END)
        self.start_server()  # Restart server to accept new connections
        threading.Thread(target=self.broadcast_hello).start()  # Resume broadcasting "hello" messages

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    master = MasterNode()
    master.run()
