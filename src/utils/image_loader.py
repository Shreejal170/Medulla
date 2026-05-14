import base64
def load_image(file_path:str)-> str:
        """Utility function to load an image file and convert it to a base64-encoded string."""
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    