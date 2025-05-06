import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from loguru import logger


def load_raw_json(input_path: str) -> List[Dict]:
    """Load and validate the raw input JSON file"""
    try:
        with open(input_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Raw input JSON must be a list of image data")

        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {str(e)}")


def convert_to_siglip_format(raw_data: List[Dict], image_dir: str) -> Dict:
    """Convert raw input format to SigLIP input format"""
    siglip_data = {"images": []}

    for item in raw_data:
        # Extract required fields
        uid = item.get("id")
        image_filename = item.get("image")

        if not uid or not image_filename:
            logger.warning(f"Skipping item with missing required fields: {item}")
            continue

        # Construct full image path
        image_path = os.path.join(image_dir, image_filename)

        # Verify image exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        # Create SigLIP format entry
        siglip_entry = {
            "uid": uid,
            "image_path": image_path,
            "metadata": {
                "original_filename": image_filename,
                "conversations": item.get("conversatons", []),  # Note: keeping original typo in field name
            },
        }

        siglip_data["images"].append(siglip_entry)

    return siglip_data


def save_siglip_json(data: Dict, output_path: str):
    """Save data in SigLIP JSON format"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Successfully saved SigLIP JSON to {output_path}")
    except Exception as e:
        raise ValueError(f"Error saving SigLIP JSON: {str(e)}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert raw input JSON to SigLIP format")
    parser.add_argument("--input_json", type=str, required=True, help="Path to raw input JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save SigLIP format JSON")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    try:
        # Load raw input JSON
        logger.info(f"Loading raw input JSON from {args.input_json}")
        raw_data = load_raw_json(args.input_json)
        logger.info(f"Loaded {len(raw_data)} images from raw input")

        # Convert to SigLIP format
        logger.info("Converting to SigLIP format...")
        siglip_data = convert_to_siglip_format(raw_data, args.image_dir)
        logger.info(f"Converted {len(siglip_data['images'])} images to SigLIP format")

        # Save SigLIP JSON
        save_siglip_json(siglip_data, args.output_json)

    except Exception as e:
        logger.error(f"Error in conversion process: {e}")
        raise


if __name__ == "__main__":
    main()
