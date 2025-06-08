import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks
import tkinter as tk
from tkinter import filedialog
import os

class AudioFFTAnalyzer:
    def __init__(self):
        self.audio = None
        self.sample_rate = None
        self.samples = None
        
    def load_mp3(self, file_path=None):
        """Load MP3 file, either from provided path or via file dialog"""
        if file_path is None:
            # Create a simple file dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            file_path = filedialog.askopenfilename(
                title="Select MP3 file",
                filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
            )
            root.destroy()
            
        if not file_path:
            print("No file selected.")
            return False
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        try:
            # Load the MP3 file
            self.audio = AudioSegment.from_mp3(file_path)
            self.sample_rate = self.audio.frame_rate
            print(f"Loaded: {file_path}")
            print(f"Duration: {len(self.audio) / 1000.0:.2f} seconds")
            print(f"Sample rate: {self.sample_rate} Hz")
            print(f"Channels: {self.audio.channels}")
            return True
        except Exception as e:
            print(f"Error loading MP3 file: {e}")
            return False
    
    def resample_audio(self, target_sample_rate=1000):
        """Resample audio to target sample rate (default 1000 Hz for 1ms sampling)"""
        if self.audio is None:
            print("No audio loaded. Please load an MP3 file first.")
            return False
            
        try:
            # Convert to mono if stereo
            if self.audio.channels > 1:
                self.audio = self.audio.set_channels(1)
                
            # Resample to target sample rate
            resampled_audio = self.audio.set_frame_rate(target_sample_rate)
            
            # Convert to numpy array
            self.samples = np.array(resampled_audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            if resampled_audio.sample_width == 2:  # 16-bit
                self.samples = self.samples / 32768.0
            elif resampled_audio.sample_width == 4:  # 32-bit
                self.samples = self.samples / 2147483648.0
                
            self.sample_rate = target_sample_rate
            
            print(f"Resampled to {target_sample_rate} Hz")
            print(f"Total samples: {len(self.samples)}")
            print(f"Duration after resampling: {len(self.samples) / target_sample_rate:.2f} seconds")
            
            return True
        except Exception as e:
            print(f"Error resampling audio: {e}")
            return False
    
    def perform_fft(self, window_size=1024):
        """Perform FFT analysis on the audio samples"""
        if self.samples is None:
            print("No samples available. Please load and resample audio first.")
            return None, None
            
        try:
            # Apply window function to reduce spectral leakage
            window = np.hanning(window_size)
            
            # Calculate number of windows
            num_windows = len(self.samples) // window_size
            
            if num_windows == 0:
                print("Audio too short for FFT analysis")
                return None, None
            
            # Prepare arrays for FFT results
            fft_results = []
            
            # Process audio in windows
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                # Extract window and apply window function
                windowed_samples = self.samples[start_idx:end_idx] * window
                
                # Perform FFT
                fft_result = np.fft.fft(windowed_samples)
                fft_magnitude = np.abs(fft_result)
                
                fft_results.append(fft_magnitude)
            
            # Average all FFT results
            avg_fft = np.mean(fft_results, axis=0)
            
            # Create frequency axis
            frequencies = np.fft.fftfreq(window_size, 1/self.sample_rate)
            
            # Take only positive frequencies (first half)
            pos_frequencies = frequencies[:window_size//2]
            pos_fft = avg_fft[:window_size//2]
            
            print(f"FFT analysis completed")
            print(f"Frequency resolution: {self.sample_rate/window_size:.2f} Hz")
            print(f"Maximum frequency: {pos_frequencies[-1]:.2f} Hz")
            
            return pos_frequencies, pos_fft
            
        except Exception as e:
            print(f"Error performing FFT: {e}")
            return None, None
    
    def display_frequency_spectrum(self, frequencies, fft_magnitude, save_plot=False):
        """Display the frequency spectrum"""
        if frequencies is None or fft_magnitude is None:
            print("No FFT data to display")
            return
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Linear scale
            plt.subplot(2, 1, 1)
            plt.plot(frequencies, fft_magnitude)
            plt.title('Frequency Spectrum (Linear Scale)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Log scale
            plt.subplot(2, 1, 2)
            plt.semilogy(frequencies, fft_magnitude)
            plt.title('Frequency Spectrum (Log Scale)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (log scale)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Find dominant frequencies
            dominant_indices = np.argsort(fft_magnitude)[-5:][::-1]  # Top 5 frequencies
            print("\nDominant frequencies:")
            for i, idx in enumerate(dominant_indices):
                freq = frequencies[idx]
                magnitude = fft_magnitude[idx]
                print(f"{i+1}. {freq:.2f} Hz (magnitude: {magnitude:.2f})")
            
            if save_plot:
                plt.savefig('frequency_spectrum.png', dpi=300, bbox_inches='tight')
                print("Plot saved as 'frequency_spectrum.png'")
            
            plt.show()
            
        except Exception as e:
            print(f"Error displaying frequency spectrum: {e}")
    
    def analyze_mp3(self, file_path=None, target_sample_rate=1000, window_size=1024, save_plot=False):
        """Complete analysis pipeline"""
        print("=== MP3 Audio FFT Analyzer ===\n")
        
        # Load MP3 file
        if not self.load_mp3(file_path):
            return
        
        print()
        
        # Resample audio
        if not self.resample_audio(target_sample_rate):
            return
        
        print()
        
        # Perform FFT
        frequencies, fft_magnitude = self.perform_fft(window_size)
        if frequencies is None:
            return
        
        print()
        
        # Display results
        self.display_frequency_spectrum(frequencies, fft_magnitude, save_plot)

def main():
    """Main function to run the analyzer"""
    analyzer = AudioFFTAnalyzer()
    
    # You can specify a file path here, or leave None to use file dialog
    file_path = None  # Change this to your MP3 file path if desired
    
    # Run analysis with 1ms sampling (1000 Hz)
    analyzer.analyze_mp3(
        file_path=file_path,
        target_sample_rate=1000,  # 1ms sampling interval
        window_size=1024,
        save_plot=True  # Set to True to save the plot as PNG
    )

if __name__ == "__main__":
    main() 