import Levenshtein
from Database import fetch_medicines
    
# Step 2: Calculate the Levenshtein distance for multi-word input
def calculate_distances(input_name, medicines):
    input_words = input_name.split()  # Split input into words
    distances = []
    
    for medicine in medicines:
        id,name = medicine
        name_words = name.split()  # Split medicine name into words
        total_distance = 0
        
        # Calculate total Levenshtein distance for each word in input against each word in medicine name
        for input_word in input_words:
            min_word_distance = min(Levenshtein.distance(input_word, med_word) for med_word in name_words)
            total_distance += min_word_distance
        
        distances.append(([id,name], total_distance))
    
    return distances

# Step 3: Find the nearest medicine name
def find_nearest_medicine(input_name, medicines):
    distances = calculate_distances(input_name.lower(), [[medicine[0],medicine[1].lower()] for medicine in medicines])
    nearest_medicine = min(distances, key=lambda x: x[1])[0]  # Find the name with the smallest total distance
    return nearest_medicine

# Step 4: Main function to run the code
def SearchMedicine(input_name):
    medicines = fetch_medicines()
    nearest_medicine = find_nearest_medicine(input_name, medicines)
    print(f"The nearest medicine name to '{input_name}' is '{nearest_medicine}'")
    return nearest_medicine

if __name__ == "__main__":
    SearchMedicine("dilicate")