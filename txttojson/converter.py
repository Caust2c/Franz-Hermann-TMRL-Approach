import json
import matplotlib.pyplot as plt

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
    filepath = r"filepath to your .txt"

    track_data = parse_centerline_text(filepath)

    json_name = filepath.rsplit('.', 1)[0] + ".json"
    with open(json_name, 'w') as f:
        json.dump(track_data, f, indent=2)

    xs, ys = zip(*track_data["layout"])
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, '-o', markersize=3)
    plt.axis('equal')
    plt.title("Centerline Layout")
    plt.grid(True)
    plt.show()
