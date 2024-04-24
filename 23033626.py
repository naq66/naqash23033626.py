import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_and_process_data(file_name, year):
    """
    Extract data from the file and preprocess based on the specified year.
    """
    data_frame = pd.read_csv(file_name, header=None, delim_whitespace=True)
    if year == 2020:
        processed_data = np.hstack([np.linspace(start, stop, num=count, endpoint=False)
                                    for start, stop, count in data_frame.to_numpy()])
    else:
        processed_data = data_frame[0].values
    return processed_data

def calculate_statistics(grades):
    mean_val = np.mean(grades)
    median_val = np.median(grades)
    std_dev_val = np.std(grades, ddof=1)
    return mean_val, median_val, std_dev_val

def calculate_proportion(grades, threshold=70):
    above_threshold = np.mean(grades > threshold)
    return above_threshold

def create_enhanced_histogram(data_2020, data_2024, student_identifier):
    plt.figure(figsize=(14, 7))
    # Define bins using the range from both datasets
    bins = np.arange(min(data_2020.min(), data_2024.min()), max(data_2020.max(), data_2024.max()) + 1, 1)
    plt.hist([data_2020, data_2024], bins=bins, color=['#007ACC', '#CC007A'], alpha=0.75, label=['2020 Grades', '2024 Grades'])
    
    stats_2020 = calculate_statistics(data_2020)
    stats_2024 = calculate_statistics(data_2024)
    prop_above_2020 = calculate_proportion(data_2020)
    
    stats_info = (f'Student ID: {student_identifier}\n'
                  f'2020 - Mean: {stats_2020[0]:.2f}, Median: {stats_2020[1]:.2f}, SD: {stats_2020[2]:.2f}\n'
                  f'2024 - Mean: {stats_2024[0]:.2f}, Median: {stats_2024[1]:.2f}, SD: {stats_2024[2]:.2f}\n'
                  f'Proportion > 70 (2020): {prop_above_2020:.2%}')
    
    plt.gca().text(0.02, 0.98, stats_info, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5))
    
    plt.title('Inter-Year Grade Distribution Dynamics', fontsize=18, fontweight='bold', color='#444444')
    plt.xlabel('Grade Scores', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_filename = f'{student_identifier}.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()

    return output_filename

unique_student_id = "23033626"
processed_grades_2020 = fetch_and_process_data('2020input6.csv', 2020)
processed_grades_2024 = fetch_and_process_data('2024input6.csv', 2024)

histogram_filename = create_enhanced_histogram(processed_grades_2020, processed_grades_2024, unique_student_id)
print(f"Histogram saved as {histogram_filename}.")
