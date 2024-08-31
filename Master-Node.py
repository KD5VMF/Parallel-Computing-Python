import socket
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MasterNode:
    def __init__(self, host="0.0.0.0", port=5000, broadcast_port=5001):
        self.host = host
        self.port = port
        self.broadcast_port = broadcast_port
        self.slave_connections = []
        self.slave_performance = []
        self.broadcasting = True
        self.running = True
        self.window = tk.Tk()
        self.window.title("Master Node")
        self.window.geometry("600x400")
        self.window.resizable(False, False)
        self.center_window()

        # Create main frame
        main_frame = tk.Frame(self.window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Server info section
        server_frame = tk.LabelFrame(main_frame, text="Server Info", padx=10, pady=10, font=("Arial", 12))
        server_frame.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(server_frame, text=f"IP: {self.get_ip_address()} | Port: {self.port}", font=("Arial", 12))
        self.info_label.pack(anchor="w")

        # Connected slaves section
        connection_frame = tk.LabelFrame(main_frame, text="Connected Slaves", padx=10, pady=10, font=("Arial", 12))
        connection_frame.pack(fill=tk.BOTH, expand=True)

        self.connection_listbox = tk.Listbox(connection_frame, height=5, font=("Arial", 12))
        self.connection_listbox.pack(fill=tk.BOTH, expand=True)

        # Matrix size and control buttons
        control_frame = tk.Frame(main_frame, padx=10, pady=10)
        control_frame.pack(fill=tk.BOTH, expand=True)

        matrix_size_label = tk.Label(control_frame, text="Matrix Size (NxN):", font=("Arial", 12))
        matrix_size_label.grid(row=0, column=0, sticky="w")

        self.matrix_size_var = tk.StringVar(value="1024x1024")
        self.matrix_size_combobox = ttk.Combobox(control_frame, textvariable=self.matrix_size_var, state="readonly", font=("Arial", 12))
        self.matrix_size_combobox['values'] = self.generate_matrix_options()
        self.matrix_size_combobox.grid(row=0, column=1, sticky="e")

        self.ready_button = tk.Button(control_frame, text="Ready", command=self.ready, font=("Arial", 12))
        self.ready_button.grid(row=1, column=0, pady=10)

        self.go_button = tk.Button(control_frame, text="Go", command=self.start_computation_thread, font=("Arial", 12))
        self.go_button.grid(row=1, column=1, pady=10)
        self.go_button.config(state=tk.DISABLED)

        self.server_socket = None
        self.accept_connections_thread = None

        # Start the server and broadcast
        self.start_server()
        self.broadcast_hello()

        # Bind keys for exit
        self.window.bind('<F8>', self.exit_program)
        self.window.bind('<Control-x>', self.exit_program)

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def generate_matrix_options(self):
        sizes = [
            "2x2", "4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024",
            "2048x2048", "3072x3072", "4096x4096", "5120x5120", "6144x6144", "7168x7168", "8192x8192",
            "12288x12288", "16384x16384", "20480x20480", "24576x24576", "32768x32768", "65536x65536",
            "131072x131072", "257536x257536"
        ]
        return sizes

    def exit_program(self, event=None):
        self.broadcasting = False
        self.running = False
        self.window.destroy()
        for conn in self.slave_connections:
            conn.close()
        if self.server_socket:
            self.server_socket.close()
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
        if self.server_socket:
            self.server_socket.close()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Master Node is listening on {self.host}:{self.port}")

        # Start the thread to accept connections
        self.accept_connections_thread = threading.Thread(target=self.accept_connections)
        self.accept_connections_thread.start()

    def accept_connections(self):
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                self.slave_connections.append(client_socket)
                self.connection_listbox.insert(tk.END, f"Connected: {address}")
                print(f"New Slave connected from {address}")
            except OSError:
                break

    def broadcast_hello(self):
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        message = f"{self.get_ip_address()}:{self.port}"
        if self.broadcasting:
            broadcast_socket.sendto(message.encode(), ('<broadcast>', self.broadcast_port))
            print(f"Broadcasting hello: {message}")
        self.window.after(2000, self.broadcast_hello)  # Broadcast every 2 seconds

    def ready(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
        else:
            self.broadcasting = False
            self.ready_button.config(state=tk.DISABLED)
            self.go_button.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Ready to start the computation!")
            print("Master is ready to start computation.")

    def start_computation_thread(self):
        # Disable the "Go" button to prevent further clicks
        self.go_button.config(state=tk.DISABLED)
        
        # Start the computation in a separate thread
        computation_thread = threading.Thread(target=self.start_computation)
        computation_thread.start()

    def start_computation(self):
        if len(self.slave_connections) == 0:
            messagebox.showwarning("Warning", "No Slaves connected!")
            return

        print("Starting computation...")
        matrix_size = self.matrix_size_var.get()
        rows, cols = map(int, matrix_size.split("x"))
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)

        num_slaves = len(self.slave_connections)
        sub_matrix_size = rows // num_slaves
        extra_rows = rows % num_slaves
        start_row = 0

        self.slave_performance.clear()

        try:
            threads = []
            for i, conn in enumerate(self.slave_connections):
                rows_to_send = sub_matrix_size + (1 if i < extra_rows else 0)
                sub_matrix_a = matrix_a[start_row:start_row + rows_to_send, :]
                start_row += rows_to_send

                print(f"Sending data to Slave {i + 1} at {conn.getpeername()}...")
                thread = threading.Thread(target=self.send_data_to_slave, args=(conn, sub_matrix_a, matrix_b, rows_to_send))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            results = []
            for performance in self.slave_performance:
                results.append(performance['result'])

            if results:
                final_result = np.vstack(results)
                print("All results received. Displaying results...")
                self.show_results(final_result)
            else:
                print("No results received from Slaves.")
                messagebox.showerror("Error", "No results received from Slaves.")

        except Exception as e:
            print(f"Error during computation: {e}")
            messagebox.showerror("Error", f"An error occurred during computation: {e}")
        finally:
            self.reset_master()

    def send_data_to_slave(self, conn, sub_matrix_a, matrix_b, rows_to_send):
        sub_matrix_shape = sub_matrix_a.shape
        matrix_b_shape = matrix_b.shape

        try:
            conn.sendall(f"{sub_matrix_shape[0]},{sub_matrix_shape[1]},{matrix_b_shape[0]},{matrix_b_shape[1]}".encode())
            conn.recv(1)  # Acknowledgement

            conn.sendall(sub_matrix_a.tobytes())
            conn.sendall(matrix_b.tobytes())

            print(f"Waiting for result from Slave at {conn.getpeername()}...")
            result = self.recv_exact(conn, 8 * rows_to_send * matrix_b_shape[1])
            result_matrix = np.frombuffer(result, dtype=np.float64).reshape(rows_to_send, matrix_b_shape[1])

            time_data = self.recv_exact(conn, 8)
            computation_time = np.frombuffer(time_data, dtype=np.float64)[0]

            self.slave_performance.append({
                'address': conn.getpeername(),
                'rows': rows_to_send,
                'result': result_matrix,
                'computation_time': computation_time
            })

            print(f"Result received from Slave at {conn.getpeername()}. Time taken: {computation_time:.4f} seconds.")

        except Exception as e:
            print(f"Error sending data to Slave {conn.getpeername()}: {e}")

    def recv_exact(self, conn, size):
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
        result_window.geometry("800x600")
        result_window.state('zoomed')  # Maximize window

        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create first tab for result summary
        summary_frame = tk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")

        result_label = tk.Label(summary_frame, text="Matrix Multiplication Results", font=("Arial", 14))
        result_label.pack(pady=10)

        columns = ("Address", "Rows Processed", "Time Taken (s)")
        tree = ttk.Treeview(summary_frame, columns=columns, show="headings")
        tree.heading("Address", text="PC Address")
        tree.heading("Rows Processed", text="Rows Processed")
        tree.heading("Time Taken (s)", text="Time Taken (s)")

        for performance in self.slave_performance:
            tree.insert("", "end", values=(
                performance['address'],
                performance['rows'],
                f"{performance['computation_time']:.4f}"
            ))

        tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        button_frame = tk.Frame(summary_frame)
        button_frame.pack(fill=tk.X, pady=10)

        save_button = tk.Button(button_frame, text="Save Results", command=lambda: self.save_results(final_result))
        save_button.pack(side=tk.LEFT, padx=10)

        close_button = tk.Button(button_frame, text="Close", command=lambda: self.close_results_window(result_window))
        close_button.pack(side=tk.RIGHT, padx=10)

        # Create second tab for 2D Heatmap
        visualization_frame = tk.Frame(notebook)
        notebook.add(visualization_frame, text="2D Heatmap")

        fig, ax = plt.subplots()

        # Plotting the 2D heatmap with an enhanced colormap
        cax = ax.matshow(final_result, cmap='cividis')  # You can try 'plasma', 'coolwarm', or 'cividis'

        # Adding gridlines to make it easier on the eyes
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)

        # Enhancing the color bar
        cbar = fig.colorbar(cax)
        cbar.set_label('Matrix Values', rotation=270, labelpad=15)

        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')

        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self, final_result):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(np.array2string(final_result, separator=', '))
            messagebox.showinfo("Saved", f"Results saved to {file_path}")

    def close_results_window(self, window):
        window.destroy()
        self.reset_master()

    def reset_master(self):
        print("Resetting Master Node for new connections...")
        for conn in self.slave_connections:
            conn.close()
        self.slave_connections.clear()
        self.connection_listbox.delete(0, tk.END)
        self.ready_button.config(state=tk.NORMAL)
        self.go_button.config(state=tk.DISABLED)
        self.broadcasting = True
        self.start_server()  # Restart server to accept new connections
        self.broadcast_hello()
        messagebox.showinfo("Ready", "The system is ready for new connections.")
        print("Master Node is ready for new connections.")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    master = MasterNode()
    master.run()
