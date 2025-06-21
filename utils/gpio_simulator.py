"""
Simulates GPIO control for machine stop/start (replace with RPi.GPIO for hardware).
Provides mock pin logic for easy hardware replacement.
"""

class GPIOSimulator:
    def __init__(self):
        self.machine_running = True
        self.pins = {}  # Simulate GPIO pins (e.g., {17: 0})
        self.machine_relay_pin = 17  # Example relay pin
        self.pins[self.machine_relay_pin] = 1  # 1=ON (machine running), 0=OFF (stopped)

    def set_pin(self, pin, value):
        self.pins[pin] = value
        print(f"[GPIO SIM] Pin {pin} set to {value}")

    def get_pin(self, pin):
        return self.pins.get(pin, 0)

    def stop_machine(self):
        self.machine_running = False
        self.set_pin(self.machine_relay_pin, 0)
        print("[GPIO SIM] Machine stopped! (Relay OFF)")

    def start_machine(self):
        self.machine_running = True
        self.set_pin(self.machine_relay_pin, 1)
        print("[GPIO SIM] Machine started! (Relay ON)")

    def get_status(self):
        return "Running" if self.machine_running else "Auto-stopped" 