import streamlit as st


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


if __name__ == '__main__':
    main()
