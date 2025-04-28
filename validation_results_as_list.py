import json
"""
paths = [
    "build/validate_average/best-model-1-5/1744652913/averages.json",
    "build/validate_average/best-model-1-10/1744653012/averages.json",
    "build/validate_average/best-model-1-15/1744653117/averages.json",
    "build/validate_average/best-model-1-20/1744653210/averages.json",
    "build/validate_average/best-model-1-50/1744653271/averages.json"]

paths = ["build/validate_average/best-model-1-5/1744652913/std.json",
    "build/validate_average/best-model-1-10/1744653012/std.json",
    "build/validate_average/best-model-1-15/1744653117/std.json",
    "build/validate_average/best-model-1-20/1744653210/std.json",
    "build/validate_average/best-model-1-50/1744653271/std.json"]
s

paths = ["assets/results-soft-start/5/1744567436/averages.json", 
         "assets/results-soft-start/10/1744567460/averages.json",
         "assets/results-soft-start/15/1744567485/averages.json",
         "assets/results-soft-start/20/1744567508/averages.json",
         "assets/results-soft-start/50/1744567525/averages.json",
         "assets/results-soft-start/80/1744567533/averages.json"]
         """

paths = ["assets/results-soft-start/5/1744567436/std.json", 
         "assets/results-soft-start/10/1744567460/std.json",
         "assets/results-soft-start/15/1744567485/std.json",
         "assets/results-soft-start/20/1744567508/std.json",
         "assets/results-soft-start/50/1744567525/std.json",
         "assets/results-soft-start/80/1744567533/std.json"]

# Dictionary to store lists for each key
results_by_key = {}

for json_file_path in paths:
    # Read and parse the JSON file
    with open(json_file_path, "r") as file:
        validation_results = json.load(file)

    for key, value in validation_results.items():
        if key not in results_by_key:
            results_by_key[key] = []
        results_by_key[key].append(value)

# Print the lists for each key
for key, values in results_by_key.items():
    print(f"{key}: {values}")
