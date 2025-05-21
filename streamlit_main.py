import streamlit as st
from PIL import Image
import numpy as np


def main():
    st.title("Streamlit demo for Aplied Machine Learning: Depth prediction.")
    st.divider()
    intro_paragraph = """
    This is a stream lit demo for the AML project of group 16.
    The project is a depth estimating model from a RGB image.

    In this demo you will be allowed to upload an image covert it using our
    model and download the resulting detph image from it.
    """
    st.markdown(intro_paragraph)
    st.divider()
    rgb_image = None
    depth_output = None
    upload_text = """
    Here you will upload an RGb 8-bit colour image for depth estimation.
    """
    st.markdown(upload_text)
    uploaded_image = st.file_uploader(label="Upload image",
                                      type=[".png", ".jpeg"])

    if uploaded_image:
        rgb_image = np.array(Image.open(uploaded_image))
        image_size = rgb_image.shape
        st.image(rgb_image)
        st.write(f"height: {image_size[0]}, width: {image_size[1]}, " +
                 f"channels: {image_size[2]}")

        if image_size[2] > 3:
            st.error("We do not accept RGBa images.")

    run_model = st.button("Start conversion.")

    if run_model and rgb_image.any():
        st.write("work in progress")

        depth_output = np.random.random((image_size[0], image_size[1], 1))

    if depth_output is not None and depth_output.any():
        st.image(depth_output)

    prepare_export = st.button("prepare depth map export")

    if prepare_export and depth_output.any():
        pass


if __name__ == '__main__':
    main()
