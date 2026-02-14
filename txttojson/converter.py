import json
import matplotlib.pyplot as plt
import os

def parse_centerline_text(file_path, track_width=32.0):
    layout = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
            parts = line.replace(',', '.').split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    layout.append([x, y])
                except ValueError:
                    continue
    return {
        "layout": layout,
        "width": round(track_width, 2)
    }

if __name__ == "__main__":
    filepath = input("Enter path to .txt file: ").strip().strip('"')

    if not os.path.isfile(filepath):
        print("File not found.")
        exit(1)

    track_data = parse_centerline_text(filepath)

    json_name = filepath.rsplit('.', 1)[0] + ".json"
    with open(json_name, 'w') as f:
        json.dump(track_data, f, indent=2)

    print(f"JSON saved to: {json_name}")

    if track_data["layout"]:
        xs, ys = zip(*track_data["layout"])
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, '-o', markersize=3)
        plt.axis('equal')
        plt.title("Centerline Layout")
        plt.grid(True)
        plt.show()
    else:
        print("No valid layout points found.")
