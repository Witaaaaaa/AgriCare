import os

# Define the expected structure
expected_structure = {
    "yolov5": ["best.pt"],
    "static": {
        "uploads": {
            "results": []
        },
        "assets": {
            "css": [],
            "js": ["script.js"],
            "img": ["favicon.png", "icon.png", "logo.png"],
            "vendor": ["bootstrap", "aos", "swiper", "glightbox"]
        }
    },
    "templates": ["index.html", "check.html", "about.html", "contact.html"]
}

# Base directory to check
base_dir = r"d:\SKRIPSI\AgriCare"

def check_structure(base_path, structure):
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        if isinstance(value, dict):
            # Check if folder exists
            if not os.path.isdir(path):
                print(f"Missing folder: {path}")
            else:
                check_structure(path, value)
        elif isinstance(value, list):
            # Check if folder exists
            if not os.path.isdir(path):
                print(f"Missing folder: {path}")
            else:
                # Check files in the folder
                for file in value:
                    file_path = os.path.join(path, file)
                    if not os.path.isfile(file_path):
                        print(f"Missing file: {file_path}")

# Run the check
check_structure(base_dir, expected_structure)
