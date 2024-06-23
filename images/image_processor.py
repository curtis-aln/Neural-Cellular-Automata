from PIL import Image
import os

def compress_and_convert_to_bw(image_path, compress_factor):
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Calculate the new dimensions based on compress factor
    new_width = original_width // compress_factor
    new_height = original_height // compress_factor

    print(f"Original dimensions: {original_width}x{original_height} ({original_width * original_height:,} total)")
    print(f"New dimensions: {new_width}x{new_height} ({new_width * new_height:,} total)")
    print(f"compression factor of {compress_factor}")

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    black_white_img = resized_image.convert('L')

    og_filename, filetype = os.path.splitext(image_path)
    new_filename = f"{og_filename}_compressed{filetype}"
    black_white_img.save(new_filename)

    print(f"[NOTE]: Compressed and converted image saved as {new_filename}")


if __name__ == "__main__":
    image_folder = "images"
    image_name = "astro.jpg"
    image_path = os.path.join(image_folder, image_name)
    compress_factor = 50  # Change this value to adjust compression

    compress_and_convert_to_bw(image_path, compress_factor)
