from utils.matching_fingerprint import load_minutiae_from_json

if __name__ == '__main__':
    minutiae_data = load_minutiae_from_json("minutiae_feature.json")
    
    for image_id, features in minutiae_data.items():
        print(f"\n📌 {image_id} có {len(features)} minutiae:")
        
        # for i, feature in enumerate(features, 1):
        #     x = feature["x"]
        #     y = feature["y"]
        #     print(f"  {i:02d}. (x={x}, y={y}) - type: {feature['type']}")