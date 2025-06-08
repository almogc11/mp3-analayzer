from audio_fft_analyzer import AudioFFTAnalyzer

def example_basic_usage():
    """Basic usage example - uses file dialog to select MP3"""
    analyzer = AudioFFTAnalyzer()
    
    # This will open a file dialog to select an MP3 file
    analyzer.analyze_mp3(
        target_sample_rate=1000,  # 1ms sampling interval as requested
        window_size=1024,
        save_plot=True
    )

def example_with_specific_file():
    """Example with a specific file path"""
    analyzer = AudioFFTAnalyzer()
    
    # Replace with your actual MP3 file path
    mp3_file = "My Way.mp3"
    
    analyzer.analyze_mp3(
        file_path=mp3_file,
        target_sample_rate=1000,  # 1ms sampling interval
        window_size=1024,
        save_plot=True
    )

def example_step_by_step():
    """Example showing step-by-step analysis"""
    analyzer = AudioFFTAnalyzer()
    
    # Step 1: Load MP3 file
    if analyzer.load_mp3():  # Will open file dialog
        print("✓ MP3 file loaded successfully")
        
        # Step 2: Resample to 1000 Hz (1ms intervals)
        if analyzer.resample_audio(1000):
            print("✓ Audio resampled to 1000 Hz")
            
            # Step 3: Perform FFT analysis
            frequencies, magnitudes = analyzer.perform_fft(1024)
            if frequencies is not None:
                print("✓ FFT analysis completed")
                
                # Step 4: Display results
                analyzer.display_frequency_spectrum(frequencies, magnitudes, save_plot=True)
                print("✓ Frequency spectrum displayed")

def example_high_resolution():
    """Example with higher frequency resolution"""
    analyzer = AudioFFTAnalyzer()
    
    # Use higher sampling rate for better frequency resolution
    # But note: this goes against the 1ms requirement
    analyzer.analyze_mp3(
        target_sample_rate=44100,  # Standard audio sampling rate
        window_size=4096,  # Larger window for better frequency resolution
        save_plot=True
    )

if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Basic usage (file dialog)")
    print("2. With specific file path")
    print("3. Step-by-step analysis")
    print("4. High resolution analysis")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_with_specific_file()
    elif choice == "3":
        example_step_by_step()
    elif choice == "4":
        example_high_resolution()
    else:
        print("Invalid choice. Running basic usage example...")
        example_basic_usage() 