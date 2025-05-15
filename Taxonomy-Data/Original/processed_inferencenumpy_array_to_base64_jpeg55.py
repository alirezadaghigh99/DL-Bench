def numpy_array_to_base64_jpeg(
    image: np.ndarray,
) -> Union[str]:
    _, img_encoded = cv2.imencode(".jpg", image)
    image_bytes = np.array(img_encoded).tobytes()
    return encode_base_64(payload=image_bytes)