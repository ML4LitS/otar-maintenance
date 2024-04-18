import os
import re
import statistics
from datetime import datetime, timedelta

def parse_datetime(datetime_str):
    # Extended regex to capture the entire datetime string correctly, including the year
    match = re.search(r'(\bMon|\bTue|\bWed|\bThu|\bFri|\bSat|\bSun)(.*)(\d{4})', datetime_str)
    if match:
        datetime_part = match.group(0)
        # Attempt to identify and remove the time zone abbreviation manually
        datetime_part_no_tz = re.sub(r'GMT|BST', '', datetime_part).strip()
        try:
            # Parse datetime without the time zone abbreviation
            parsed_datetime = datetime.strptime(datetime_part_no_tz, '%a %d %b %H:%M:%S %Y')
            # Check for BST and adjust the time if necessary
            if 'BST' in datetime_str:
                # Assuming BST is GMT+1
                parsed_datetime -= timedelta(hours=1)
            return parsed_datetime
        except ValueError as e:
            print(f"Error parsing date: '{datetime_str}'. Extracted part: '{datetime_part_no_tz}'. Error: {e}")
    return None


def parse_log_file(file_path):
    patterns = {
        'Start Sentenciser': None,
        'End Sentenciser': None,
        'Start CleanTags': None,
        'End CleanTags': None,
        'Start Bioformer ML processing': None,
        'End Bioformer ML processing': None,
        'Start gsutil copy': None,
        'End gsutil copy': None,
        'Start Section Tagger': None,  # Only for fulltext
        'End Section Tagger': None,  # Only for fulltext
    }
    with open(file_path) as f:
        for line in f:
            for key in patterns.keys():
                if key in line:
                    # Extract the date and time from the line, ignoring the time zone for simplicity
                    date_time_str = " ".join(line.split()[2:])
                    datetime_parsed = parse_datetime(date_time_str)
                    if datetime_parsed:
                        patterns[key] = datetime_parsed

    # Calculate time taken for each process in hours
    results = []
    for start, end in zip(list(patterns.keys())[::2], list(patterns.keys())[1::2]):
        if patterns[start] and patterns[end]:
            time_taken = (patterns[end] - patterns[start]).total_seconds() / 3600.0
            results.append((start.replace('Start ', ''), patterns[start], patterns[end], time_taken))
    return results

# Example usage
base_dir = "/hps/nobackup/literature/otar-pipeline/slurm_daily_pipeline_api"
dates = ["2024_03_23", "2024_03_24", "2024_03_27", "2024_03_28", "2024_03_29", "2024_03_30", "2024_03_31", "2024_04_01", "2024_04_02"]
sub_folders = ["abstract", "fulltext"]

# Collect data
data = []
for date in dates:
    for sub_folder in sub_folders:
        log_folder = os.path.join(base_dir, date, sub_folder, "log")
        for file in os.listdir(log_folder):
            if file.startswith("patch") and file.endswith(".out"):
                file_path = os.path.join(log_folder, file)
                for process, start_time, end_time, time_taken in parse_log_file(file_path):
                    data.append([date, sub_folder, file, process, start_time, end_time, time_taken])

# Output CSV
print("\n\n\n\n\n\nDate,SubFolder,FileName,Process,StartTime,EndTime,TimeTaken(Hours)")
for row in data:
    print(",".join(str(item) for item in row))

# Organize data by process type
process_data = {}
for _, sub_folder, _, process, _, _, time_taken in data:
    key = (sub_folder, process)
    if key not in process_data:
        process_data[key] = []
    process_data[key].append(time_taken)

# Calculate and print statistics for each process
for key, times in process_data.items():
    sub_folder, process = key
    mean_time = statistics.mean(times)
    max_time = max(times)
    try:
        std_dev = statistics.stdev(times)
    except statistics.StatisticsError:
        # This can happen if there's only one data point
        std_dev = 0
    print(f"{process} in {sub_folder}: Mean={mean_time:.2f} hours, Max={max_time:.2f} hours, StdDev={std_dev:.2f} hours")

