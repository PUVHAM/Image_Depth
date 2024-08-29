import time
import cv2
import streamlit as st

from src.pixelwise_matching import pixel_wise_matching
from src.window_based_matching import window_based_matching

@st.cache_data(max_entries=1000)
def inference_and_display_result(algo_type, 
                                 similiarity_type, 
                                 left_img_path,
                                 right_img_path, 
                                 disparity_range, 
                                 kernel_size,
                                 scale):
    if algo_type == 'Pixel-wise matching':
        depth, depth_color = pixel_wise_matching(left_img_path=left_img_path,
                                                 right_img_path=right_img_path,
                                                 similiarity_type=similiarity_type,
                                                 disparity_range=disparity_range,
                                                 scale=scale)
    elif algo_type == 'Window-based matching':
        depth, depth_color = window_based_matching(left_img_path=left_img_path,
                                                   right_img_path=right_img_path,
                                                   similiarity_type=similiarity_type,
                                                   disparity_range=disparity_range,
                                                   kernel_size=kernel_size,
                                                   scale=scale)
        
    return depth, depth_color


def main():
    st.set_page_config(
        page_title="Image Depth Estimation App",  
        page_icon=":camera:",        
        layout="wide",                            
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/PUVHAM/Image_Depth.git',  
            'Report a Bug': 'mailto:phamquangvu19082005@gmail.com',        
            'About': "# Image Depth Estimation App\n"
                    "This app demonstrates stereo matching for image depth estimation.\n"
                    "Upload stereo images to estimate the depth map and visualize the results."
        }
    )

    st.title(':frame_with_picture: :blue[Stereo Matching] Image Depth Estimation Demo')

    abbreviations = {
        "Pixel-wise matching": "PW",
        "Window-based matching": "WB"
    }

    with st.sidebar:
        st.header("Configuration")
        
        option = st.selectbox('Algorithm Type', ('Pixel-wise matching', 'Window-based matching'))
        short_option = abbreviations.get(option, option)

        if short_option == 'PW':
            similarity_type = st.selectbox('Similarity Function', ('l1', 'l2'))
            kernel_size = 0
        else:
            similarity_type = st.selectbox('Similarity Function', ('l1', 'l2', 'cosine', 'correlation'))
            kernel_size = st.slider('Kernel Size', min_value=1, max_value=15, value=3, step=2)

        img_content = st.selectbox('Image Content', ('Tsukuba', 'Aloe'))

        if img_content == 'Aloe':
            aloe_right_version = st.selectbox('Aloe Right Image Version', ('Version 1', 'Version 2', 'Version 3'))

        if short_option == 'PW':
            disparity_range = st.slider('Disparity Range', min_value=1, max_value=200, value=16)
            scale = st.slider('Scale Factor', min_value=1, max_value=200, value=16)
        else:
            disparity_range = st.slider('Disparity Range', min_value=1, max_value=200, value=64)
            scale = st.slider('Scale Factor', min_value=1, max_value=200, value=3)

        submitted = st.button('Submit')

    if img_content == 'Tsukuba':
        left_img_path = 'img/Tsukuba/left.png'
        right_img_path = 'img/Tsukuba/right.png'
    elif img_content == 'Aloe':
        left_img_path = 'img/Aloe/Aloe_left_1.png'
        if aloe_right_version == 'Version 1':
            right_img_path = 'img/Aloe/Aloe_right_1.png'
        elif aloe_right_version == 'Version 2':
            right_img_path = 'img/Aloe/Aloe_right_2.png'
        elif aloe_right_version == 'Version 3':
            right_img_path = 'img/Aloe/Aloe_right_3.png'
    else:
        raise FileNotFoundError('Image content not found!')

    if submitted:
        with st.spinner('Processing...'):
            start_time = time.time()
            depth, depth_color = inference_and_display_result(algo_type=option,
                                                              similiarity_type=similarity_type,
                                                              left_img_path=left_img_path,
                                                              right_img_path=right_img_path,
                                                              disparity_range=disparity_range,
                                                              kernel_size=kernel_size,
                                                              scale=scale)
            
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            end_time = time.time() - start_time
        
        st.divider()

        st.success(f'Processing completed in {end_time:.3f} seconds')

        st.subheader("Input Images")
        col1, col2 = st.columns(2)
        col1.image(left_img_path, caption='Left image', use_column_width=True)
        col2.image(right_img_path, caption='Right image', use_column_width=True)

        st.subheader("Results")
        col3, col4 = st.columns(2)
        col3.image(depth, caption='Disparity map', use_column_width=True)
        col4.image(depth_color, caption='Disparity map (heatmap)', use_column_width=True)

        st.download_button(
            label="Download Disparity Map",
            data=cv2.imencode('.png', depth)[1].tobytes(),
            file_name="disparity_map.png",
            mime="image/png"
        )
        
    st.divider()

    st.markdown("""
        ## How to use
        1. Select the algorithm type and parameters in the sidebar
        2. Choose the image content
        3. Click 'Submit' to process and view results
        
        ## About
        This app demonstrates stereo matching for image depth estimation.
        Upload stereo images to estimate the depth map and visualize the results.
    """)

if __name__ == '__main__':
    main()