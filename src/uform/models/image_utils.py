__all__ = ["convert_to_rgb"]


# lambda is not pickable
def convert_to_rgb(image):
    return image.convert("RGB")
