import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

def analyze_audio_fft(file_path="1000 Hz Test Tone.mp3", target_sample_rate=1000, window_size=1024):
    """Simple audio FFT analyzer"""
    
    print("=== Simple Audio FFT Analyzer ===\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        print("Please make sure the MP3 file is in the same directory as this script.")
        return
    
    print(f"Loading audio file: {file_path}")
    
    try:
        # Load the audio file using librosa
        print("Loading audio... (this may take a moment)")
        audio, original_sample_rate = librosa.load(file_path, sr=None, mono=True)
        print(f"✓ Audio loaded successfully!")
        print(f"  - Duration: {len(audio) / original_sample_rate:.2f} seconds")
        print(f"  - Original sample rate: {original_sample_rate} Hz")
        print(f"  - Total samples: {len(audio)}")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    print(f"\nResampling to {target_sample_rate} Hz (1ms intervals)...")
    
    try:
        # Resample using librosa
        if original_sample_rate != target_sample_rate:
            samples = librosa.resample(audio, orig_sr=original_sample_rate, target_sr=target_sample_rate)
        else:
            samples = audio.copy()
        
        print(f"✓ Resampling completed!")
        print(f"  - New sample rate: {target_sample_rate} Hz")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Duration: {len(samples) / target_sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return
    
    print(f"\nPerforming FFT analysis...")
    
    try:
        # Apply window function to reduce spectral leakage
        window = np.hanning(window_size)
        
        # Calculate number of windows
        num_windows = len(samples) // window_size
        
        if num_windows == 0:
            print("Error: Audio too short for FFT analysis")
            return
        
        print(f"  - Processing {num_windows} windows of {window_size} samples each")
        
        # Prepare arrays for FFT results
        fft_results = []
        
        # Process audio in windows
        for i in range(num_windows):
            if i % 100 == 0:  # Progress indicator
                print(f"  - Processing window {i+1}/{num_windows}")
                
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            # Extract window and apply window function
            windowed_samples = samples[start_idx:end_idx] * window
            
            # Perform FFT
            fft_result = np.fft.fft(windowed_samples)
            fft_magnitude = np.abs(fft_result)
            
            fft_results.append(fft_magnitude)
        
        # Average all FFT results
        avg_fft = np.mean(fft_results, axis=0)
        
        # Create frequency axis
        frequencies = np.fft.fftfreq(window_size, 1/target_sample_rate)
        
        # Take only positive frequencies (first half)
        pos_frequencies = frequencies[:window_size//2]
        pos_fft = avg_fft[:window_size//2]
        
        print(f"✓ FFT analysis completed!")
        print(f"  - Frequency resolution: {target_sample_rate/window_size:.2f} Hz")
        print(f"  - Maximum frequency: {pos_frequencies[-1]:.2f} Hz")
        
    except Exception as e:
        print(f"Error performing FFT: {e}")
        return
    
    print(f"\nDisplaying results...")
    
    try:
        # Filter to human audible range (20 Hz - 20 kHz)
        audible_mask = (pos_frequencies >= 20) & (pos_frequencies <= 20000)
        audible_frequencies = pos_frequencies[audible_mask]
        audible_fft = pos_fft[audible_mask]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Linear scale - Full range
        plt.subplot(2, 2, 1)
        plt.plot(audible_frequencies, audible_fft)
        plt.title('Human Audible Range (20 Hz - 20 kHz) - Linear Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(20, 20000)
        
        # Plot 2: Log scale - Full range
        plt.subplot(2, 2, 2)
        plt.semilogy(audible_frequencies, audible_fft)
        plt.title('Human Audible Range (20 Hz - 20 kHz) - Log Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log scale)')
        plt.grid(True, alpha=0.3)
        plt.xlim(20, 20000)
        
        # Plot 3: Focus on low frequencies (20 Hz - 2 kHz) where most music content is
        low_freq_mask = (pos_frequencies >= 20) & (pos_frequencies <= 2000)
        low_frequencies = pos_frequencies[low_freq_mask]
        low_fft = pos_fft[low_freq_mask]
        
        plt.subplot(2, 2, 3)
        plt.plot(low_frequencies, low_fft)
        plt.title('Music Fundamentals (20 Hz - 2 kHz) - Linear Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(20, 2000)
        
        # Plot 4: Focus on high frequencies (2 kHz - 20 kHz) for harmonics and details
        high_freq_mask = (pos_frequencies >= 2000) & (pos_frequencies <= 20000)
        high_frequencies = pos_frequencies[high_freq_mask]
        high_fft = pos_fft[high_freq_mask]
        
        plt.subplot(2, 2, 4)
        plt.plot(high_frequencies, high_fft)
        plt.title('Harmonics & Details (2 kHz - 20 kHz) - Linear Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(2000, 20000)
        
        plt.tight_layout()
        
        # Find dominant frequencies in audible range
        audible_dominant_indices = np.argsort(audible_fft)[-10:][::-1]  # Top 10 audible frequencies
        
        print(f"✓ Analysis complete! Here are the dominant audible frequencies:")
        for i, idx in enumerate(audible_dominant_indices):
            freq = audible_frequencies[idx]
            magnitude = audible_fft[idx]
            
            # Add frequency range description
            if freq < 250:
                freq_type = "Bass"
            elif freq < 500:
                freq_type = "Low Mid"
            elif freq < 2000:
                freq_type = "Mid"
            elif freq < 6000:
                freq_type = "High Mid"
            else:
                freq_type = "Treble"
                
            print(f"  {i+1}. {freq:.2f} Hz (magnitude: {magnitude:.2f}) - {freq_type}")
            
        print(f"\nFrequency Range Analysis:")
        print(f"  - Bass (20-250 Hz): Contains fundamental tones, drums, bass")
        print(f"  - Low Mid (250-500 Hz): Warmth, body of instruments")
        print(f"  - Mid (500-2000 Hz): Clarity, presence, vocals")
        print(f"  - High Mid (2-6 kHz): Brightness, attack, definition")
        print(f"  - Treble (6-20 kHz): Air, sparkle, harmonics")
        
        # Save the plot
        plt.savefig('frequency_spectrum.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved as 'frequency_spectrum.png'")
        
        # Show the plot
        print("✓ Displaying plot...")
        plt.show()
        
    except Exception as e:
        print(f"Error displaying results: {e}")
        return
    
    print("\n=== Analysis Complete! ===")

if __name__ == "__main__":
    # Run the analysis on the original song with 40 kHz sampling
    print("Analyzing '1000 Hz Test Tone.mp3' with 40 kHz sampling rate")
    print("This will show frequencies up to 20 kHz (full human audible range: 20 Hz - 20 kHz)\n")
    
    analyze_audio_fft(
        file_path="1000 Hz Test Tone.mp3",  # Your original song
        target_sample_rate=40000,  # 40 kHz sampling rate for human audible range
        window_size=4096  # Larger window for better frequency resolution at high sample rate
    ) 