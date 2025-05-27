import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from project_name.models.cnn import CNNBackbone
from os.path import split, join
import torch


def main() -> None:
    """
    Main function where the stream lit app runs.
    """
    if not ("device" in st.session_state):
        st.session_state["device"] = torch.device("cuda"
                                                  if torch.cuda.is_available()
                                                  else "cpu")

    if not ("model" in st.session_state):
        file_location = split(__file__)[0]
        st.session_state["model"] = CNNBackbone(pretrained=True).to(
            st.session_state["device"]
        )
        st.session_state["model"].load_state_dict(torch.load(
            join(file_location, "cnn_best.pth"), weights_only=True))
        st.session_state["model"].eval()

    st.title("Streamlit demo for Applied Machine Learning: Depth prediction.")
    st.divider()
    intro_paragraph = """
    This is a stream lit demo for the AML project of group 16.
    The project is a depth estimating model from a RGB image.

    In this demo you will be allowed to upload an image covert it using our
    model and download the resulting depth image from it.
    """
    st.markdown(intro_paragraph)
    st.divider()
    upload_text = """
    Here you will upload an RGb 8-bit colour image for depth estimation.
    """
    st.markdown(upload_text)
    st.session_state["upload_image"] = st.file_uploader(label="Upload image",
                                                        type=[".png", ".jpeg"])

    if ("upload_image" in st.session_state and
            st.session_state["upload_image"] is not None):
        rgb_image = np.array(Image.open(st.session_state["upload_image"]))
        image_size = rgb_image.shape
        st.image(rgb_image)
        st.write(f"height: {image_size[0]}, width: {image_size[1]}, " +
                 f"channels: {image_size[2]}")
        st.session_state["rgb_image"] = rgb_image

        if image_size[2] > 3:
            st.error("We do not accept RGBa images.")

    st.divider()
    if (not ("upload_image" in st.session_state) or
            st.session_state["upload_image"] is None):
        if "depth_output" in st.session_state:
            st.session_state.pop("depth_output")
        return

    run_model = st.button("Start conversion.")

    if run_model and "rgb_image" in st.session_state:
        st.write("work in progress")
        model = st.session_state["model"]
        input_image = st.session_state["rgb_image"]
        orginal_shape = list(input_image.shape)[0:2] + [1]
        input_image = np.expand_dims(input_image, axis=0)

        st.session_state["depth_output"] = model(
            torch.tensor(input_image,
                         device=st.session_state["device"],
                         dtype=torch.float).permute(0, 3, 1, 2)
            ).detach().numpy().reshape(orginal_shape)

    if "depth_output" in st.session_state:
        max = np.max(st.session_state["depth_output"])
        st.write((st.session_state["depth_output"].squeeze() / max) * 255)
        st.image((st.session_state["depth_output"] / max) * 255, clamp=True)
        depth_image_size = st.session_state["depth_output"].shape
        st.write(f"height: {depth_image_size[0]}," +
                 f" width: {depth_image_size[1]}, " +
                 f"channels: {depth_image_size[2]}")

    st.divider()
    if not ('depth_output' in st.session_state):
        return

    file_name = st.text_input("file name",
                              "depth_map")
    prepare_export = st.button("prepare depth map export")

    if file_name == "":
        st.error("empty name can not generate file")
        return

    if prepare_export and st.session_state["depth_output"].any():
        with BytesIO() as buffer:
            np.save(buffer, st.session_state["depth_output"])
            st.download_button("download depth map",
                               buffer,
                               file_name + ".npy")

    st.divider()


if __name__ == '__main__':
    main()
