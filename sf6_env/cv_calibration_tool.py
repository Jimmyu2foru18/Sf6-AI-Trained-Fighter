#!/usr/bin/env python3
"""
Computer Vision Calibration Tool for Street Fighter 6
Helps calibrate health bar detection and other CV components
"""

import cv2
import numpy as np
import mss
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json
from pathlib import Path
import time

class SF6CVCalibrator:
    """GUI tool for calibrating SF6 computer vision components"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SF6 Computer Vision Calibrator")
        self.root.geometry("1200x800")
        
        # Screen capture
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        
        # Current frame and regions
        self.current_frame = None
        self.regions = {
            'player_health': None,
            'enemy_health': None,
            'player_super': None,
            'enemy_super': None,
            'timer': None
        }
        
        # Selection state
        self.selecting_region = None
        self.selection_start = None
        
        # Setup GUI
        self.setup_gui()
        
        # Start capture loop
        self.capture_loop()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for image display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup control panel
        self.setup_control_panel(control_frame)
        
        # Setup image display
        self.setup_image_display(image_frame)
    
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        # Title
        title_label = ttk.Label(parent, text="SF6 CV Calibrator", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Capture controls
        capture_frame = ttk.LabelFrame(parent, text="Screen Capture")
        capture_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(capture_frame, text="Capture Screenshot", command=self.capture_screenshot).pack(pady=5)
        ttk.Button(capture_frame, text="Start Live Capture", command=self.start_live_capture).pack(pady=5)
        ttk.Button(capture_frame, text="Stop Live Capture", command=self.stop_live_capture).pack(pady=5)
        
        # Region selection
        region_frame = ttk.LabelFrame(parent, text="Region Selection")
        region_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.region_buttons = {}
        for region_name in self.regions.keys():
            btn = ttk.Button(region_frame, text=f"Select {region_name.replace('_', ' ').title()}", 
                           command=lambda r=region_name: self.start_region_selection(r))
            btn.pack(fill=tk.X, pady=2)
            self.region_buttons[region_name] = btn
        
        # Region info
        info_frame = ttk.LabelFrame(parent, text="Region Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.region_info = tk.Text(info_frame, height=8, width=30)
        self.region_info.pack(fill=tk.BOTH, expand=True)
        
        # Color detection
        color_frame = ttk.LabelFrame(parent, text="Color Detection")
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(color_frame, text="Test Health Detection", command=self.test_health_detection).pack(pady=2)
        ttk.Button(color_frame, text="Test Super Detection", command=self.test_super_detection).pack(pady=2)
        ttk.Button(color_frame, text="Calibrate Colors", command=self.calibrate_colors).pack(pady=2)
        
        # Save/Load
        file_frame = ttk.LabelFrame(parent, text="Configuration")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Save Configuration", command=self.save_configuration).pack(pady=2)
        ttk.Button(file_frame, text="Load Configuration", command=self.load_configuration).pack(pady=2)
        ttk.Button(file_frame, text="Export to Config File", command=self.export_to_config).pack(pady=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(parent, textvariable=self.status_var)
        status_label.pack(pady=(10, 0))
    
    def setup_image_display(self, parent):
        """Setup the image display area"""
        # Canvas for image display
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for region selection
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
    
    def capture_screenshot(self):
        """Capture a single screenshot"""
        screenshot = self.sct.grab(self.monitor)
        self.current_frame = np.array(screenshot)
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGRA2RGB)
        self.update_display()
        self.status_var.set("Screenshot captured")
    
    def start_live_capture(self):
        """Start live capture mode"""
        self.live_capture = True
        self.status_var.set("Live capture started")
    
    def stop_live_capture(self):
        """Stop live capture mode"""
        self.live_capture = False
        self.status_var.set("Live capture stopped")
    
    def capture_loop(self):
        """Main capture loop"""
        if hasattr(self, 'live_capture') and self.live_capture:
            screenshot = self.sct.grab(self.monitor)
            self.current_frame = np.array(screenshot)
            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGRA2RGB)
            self.update_display()
        
        # Schedule next capture
        self.root.after(100, self.capture_loop)  # 10 FPS
    
    def update_display(self):
        """Update the image display"""
        if self.current_frame is None:
            return
        
        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Calculate scaling
        frame_height, frame_width = self.current_frame.shape[:2]
        scale_x = canvas_width / frame_width
        scale_y = canvas_height / frame_height
        scale = min(scale_x, scale_y)
        
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(self.current_frame, (new_width, new_height))
        
        # Draw regions on frame
        display_frame = resized_frame.copy()
        for region_name, region in self.regions.items():
            if region is not None:
                x, y, w, h = region
                # Scale coordinates
                x_scaled = int(x * scale)
                y_scaled = int(y * scale)
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)
                
                # Draw rectangle
                color = (255, 0, 0) if region_name.startswith('player') else (0, 255, 0)
                cv2.rectangle(display_frame, (x_scaled, y_scaled), 
                            (x_scaled + w_scaled, y_scaled + h_scaled), color, 2)
                
                # Draw label
                cv2.putText(display_frame, region_name, (x_scaled, y_scaled - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Convert to PIL Image and display
        pil_image = Image.fromarray(display_frame)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        
        # Store scale for coordinate conversion
        self.display_scale = scale
    
    def start_region_selection(self, region_name):
        """Start selecting a region"""
        self.selecting_region = region_name
        self.status_var.set(f"Click and drag to select {region_name.replace('_', ' ')}")
    
    def on_canvas_click(self, event):
        """Handle canvas click"""
        if self.selecting_region:
            self.selection_start = (event.x, event.y)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        if self.selecting_region and self.selection_start:
            # Draw selection rectangle
            self.canvas.delete("selection")
            self.canvas.create_rectangle(
                self.selection_start[0], self.selection_start[1],
                event.x, event.y,
                outline="yellow", width=2, tags="selection"
            )
    
    def on_canvas_release(self, event):
        """Handle canvas release"""
        if self.selecting_region and self.selection_start:
            # Calculate region coordinates
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            
            # Ensure x1,y1 is top-left
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Convert to original frame coordinates
            if hasattr(self, 'display_scale'):
                x1 = int(x1 / self.display_scale)
                y1 = int(y1 / self.display_scale)
                x2 = int(x2 / self.display_scale)
                y2 = int(y2 / self.display_scale)
            
            # Store region
            w = x2 - x1
            h = y2 - y1
            self.regions[self.selecting_region] = (x1, y1, w, h)
            
            # Update display and info
            self.update_display()
            self.update_region_info()
            
            # Clear selection
            self.canvas.delete("selection")
            self.selecting_region = None
            self.status_var.set(f"Region {self.selecting_region} selected")
    
    def update_region_info(self):
        """Update the region information display"""
        self.region_info.delete(1.0, tk.END)
        
        for region_name, region in self.regions.items():
            if region is not None:
                x, y, w, h = region
                self.region_info.insert(tk.END, f"{region_name}:\n")
                self.region_info.insert(tk.END, f"  Position: ({x}, {y})\n")
                self.region_info.insert(tk.END, f"  Size: {w}x{h}\n\n")
    
    def test_health_detection(self):
        """Test health bar detection"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please capture a frame first")
            return
        
        results = []
        for region_name in ['player_health', 'enemy_health']:
            region = self.regions.get(region_name)
            if region is not None:
                percentage = self.detect_health_percentage(self.current_frame, region)
                results.append(f"{region_name}: {percentage:.1f}%")
        
        if results:
            messagebox.showinfo("Health Detection Results", "\n".join(results))
        else:
            messagebox.showwarning("Warning", "No health regions defined")
    
    def test_super_detection(self):
        """Test super meter detection"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please capture a frame first")
            return
        
        results = []
        for region_name in ['player_super', 'enemy_super']:
            region = self.regions.get(region_name)
            if region is not None:
                percentage = self.detect_super_percentage(self.current_frame, region)
                results.append(f"{region_name}: {percentage:.1f}%")
        
        if results:
            messagebox.showinfo("Super Detection Results", "\n".join(results))
        else:
            messagebox.showwarning("Warning", "No super regions defined")
    
    def detect_health_percentage(self, frame, region):
        """Detect health percentage from region"""
        x, y, w, h = region
        health_region = frame[y:y+h, x:x+w]
        
        if health_region.size == 0:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(health_region, cv2.COLOR_RGB2HSV)
        
        # Define health bar color ranges (you may need to adjust these)
        health_ranges = {
            'green': ((40, 50, 50), (80, 255, 255)),
            'yellow': ((20, 50, 50), (40, 255, 255)),
            'red': ((0, 50, 50), (20, 255, 255))
        }
        
        total_health_pixels = 0
        for color_name, (lower, upper) in health_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_health_pixels += cv2.countNonZero(mask)
        
        total_pixels = health_region.shape[0] * health_region.shape[1]
        if total_pixels > 0:
            return (total_health_pixels / total_pixels) * 100
        
        return 0.0
    
    def detect_super_percentage(self, frame, region):
        """Detect super meter percentage from region"""
        x, y, w, h = region
        super_region = frame[y:y+h, x:x+w]
        
        if super_region.size == 0:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(super_region, cv2.COLOR_RGB2HSV)
        
        # Blue/cyan range for super meter
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        super_pixels = cv2.countNonZero(mask)
        total_pixels = super_region.shape[0] * super_region.shape[1]
        
        if total_pixels > 0:
            return (super_pixels / total_pixels) * 100
        
        return 0.0
    
    def calibrate_colors(self):
        """Open color calibration window"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please capture a frame first")
            return
        
        # Create color calibration window
        color_window = tk.Toplevel(self.root)
        color_window.title("Color Calibration")
        color_window.geometry("400x600")
        
        # Add HSV sliders and preview
        # This would be a more complex implementation
        messagebox.showinfo("Info", "Color calibration feature coming soon!")
    
    def save_configuration(self):
        """Save current configuration"""
        config = {
            'regions': self.regions,
            'monitor': self.monitor
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", f"Configuration saved to {filename}")
    
    def load_configuration(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.regions = config.get('regions', {})
                self.monitor = config.get('monitor', self.monitor)
                
                self.update_display()
                self.update_region_info()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def export_to_config(self):
        """Export regions to SF6 config file format"""
        if not any(self.regions.values()):
            messagebox.showwarning("Warning", "No regions defined")
            return
        
        # Generate config file content
        config_content = f'''# SF6 Computer Vision Configuration
# Generated by SF6 CV Calibrator

# Screen capture settings
MONITOR_TOP = {self.monitor['top']}
MONITOR_LEFT = {self.monitor['left']}
MONITOR_WIDTH = {self.monitor['width']}
MONITOR_HEIGHT = {self.monitor['height']}

# Health bar regions (x, y, width, height)
'''
        
        for region_name, region in self.regions.items():
            if region is not None:
                x, y, w, h = region
                var_name = region_name.upper() + '_REGION'
                config_content += f"{var_name} = ({x}, {y}, {w}, {h})\n"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(config_content)
            messagebox.showinfo("Success", f"Configuration exported to {filename}")
    
    def run(self):
        """Run the calibrator"""
        self.root.mainloop()
        
        # Cleanup
        if hasattr(self, 'sct'):
            self.sct.close()

def main():
    """Main function"""
    print("Starting SF6 Computer Vision Calibrator...")
    print("Instructions:")
    print("1. Start Street Fighter 6 and position it on screen")
    print("2. Use 'Capture Screenshot' or 'Start Live Capture'")
    print("3. Select regions for health bars, super meters, etc.")
    print("4. Test detection and save configuration")
    
    calibrator = SF6CVCalibrator()
    calibrator.run()

if __name__ == "__main__":
    main()