## NLP App Theatre Reviews

### Import packages

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image, ImageDraw, ImageFont
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

from pathlib import Path
import base64
import time

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(
    page_title="Cookie Viz Playground", layout="wide", page_icon="./images/flask.png"
)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

    images = Image.open('images/hi-paris.png')
    st.image(images, width=200)

    st.markdown("## Cookie Viz 🍪")

    st.markdown("     ")

    st.markdown("---")

    def file_select(folder='./datasets'):
        filelist = os.listdir(folder)
        st.sidebar.markdown("OR")
        selectedfile = st.sidebar.selectbox('', filelist)
        return os.path.join(folder, selectedfile)

    st.sidebar.markdown("---")
    st.sidebar.header("Select Dataset")

    if st.sidebar.button('Upload Data'):
        data = st.sidebar.file_uploader('', type=['CSV'])
        if data is not None:
            df = pd.read_csv(data)
    else:
        filename = file_select()
        st.sidebar.info('You selected {}'.format(filename))
        if filename is not None:
            df = pd.read_csv(filename)
    st.sidebar.markdown("---")
    st.sidebar.header("Select Project Step")
    nlp_steps = st.sidebar.selectbox('', ['00 - Introduction to the Project', '01 - Dataset Presentation',
                                          '02- Basic Information', '03 - Exploratory Visualization',
                                          '04 - See information collected', '05 - How your information is processed'])
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/hi-paris) <small> Cookies Viz 1.0.0 | November 2021</small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )

    if nlp_steps == "00 - Introduction to the Project":
        st.markdown("#### 00 - Introduction to the Project")
        st.write("Over the past few years, regulators have begun to consider restricting a cookie’s"
                 " lifetime or even banning cookies altogether as a way to protect consumer privacy."
                 " Most of this debate has taken place in the absence of any quantified cost-benefit analysis."
                 " To begin to fill this gap in the discourse, we estimate the potential economic damage of"
                 " lifespan restrictions on cookies."
                 " Our analysis is based on an empirical study on cookies of **54,127** users who received about"
                 " **130** million ad impressions over **2.5**"
                 " years.")

        st.write("The goal in this app is for your to understand the data that is collected when you browse online and"
                 "how the information is used afterwards. You will be able to explore part of "
                 "the data generated in one day in the US")

        st.write("**Placeholder of the Cookie Lifecycle explanation**")
        images = Image.open('images/cookie_lifecycle.png')
        st.image(images, width=600)

    elif nlp_steps == "01 - Dataset Presentation":

        if filename == "./datasets/master_csv_segment.csv":
            st.markdown("#### 01 - Dataset Presentation")
            st.success(f"You have selected the Dataset: {filename}")
            st.write("The Log-Level Segment Feed gives you data on the segment pixel loads for all of your "
                     "network- and advertiser-level segments. Information about your 3rd-party data "
                     "providers' pixels are not included. The feed contains one row per segment load.")
            num = st.number_input('No. of Rows', 5, 10)
            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(num))
            else:
                st.dataframe(df.tail(num))
            st.markdown("---")

            snippet = f"""
    
            >>> import pandas as pd
            >>> import numpy as  as np
    
            >>> df.head(5)
            #Or
            >>> df.tail(5)
    
            """
            code_header_placeholder = st.empty()
            snippet_placeholder = st.empty()
            code_header_placeholder.header(f"**Code for the step: 00 - Show  Dataset**")
            snippet_placeholder.code(snippet)

        elif filename == "./datasets/master_csv_standard.csv":
            st.markdown("#### 01 - Dataset Presentation")
            st.success(f"You have selected the Dataset: {filename}")
            st.write("The Log-Level Standard Feed provides data on your managed publishers' and/or "
                     "your managed advertisers' "
                     "transacted impressions and the resulting clicks and conversions. The feed contains one row per "
                     "transacted impression, click, or conversion. If you use impression and "
                     "clicktrackers, the feed will also contain one row per impression tracker or clicktracker event.")
            num = st.number_input('No. of Rows', 5, 10)
            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(num))
            else:
                st.dataframe(df.tail(num))
            st.markdown("---")

            snippet = f"""
    
            >>> import pandas as pd
            >>> import numpy as  as np
    
            >>> df.head(5)
            #Or
            >>> df.tail(5)
    
            """
            code_header_placeholder = st.empty()
            snippet_placeholder = st.empty()
            code_header_placeholder.header(f"**Code for the step: 00 - Show  Dataset**")
            snippet_placeholder.code(snippet)

        else:
            st.success(f"You have selected the Dataset: {filename}")
            st.write("The Log-Level Bid Landscape Feed helps you gain insight into the demand for your inventory based"
                     " on auctions involving a randomly selected set of 1% of the users on our platform."
                     " This feed contains the top fifteen bids, with one row per bid. This includes the winning bid."
                     " Please note that the winning bid may or may not have been the highest bid due to your ad quality"
                     " or yield management settings. Each user is randomly and evenly assigned to one of the 1000"
                     " separate user groups; once assigned to a group, a user stays in that group."
                     " This log covers auctions for users with a user_group_id that is less than or equal to 10. "
                     "We track the timestamp and auction ID for each.")
            num = st.number_input('No. of Rows', 5, 10)
            head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

            if head == 'Head':
                st.dataframe(df.head(num))
            else:
                st.dataframe(df.tail(num))
            st.markdown("---")

            snippet = f"""

            >>> import pandas as pd
            >>> import numpy as  as np

            >>> df.head(5)
            #Or
            >>> df.tail(5)

            """
            code_header_placeholder = st.empty()
            snippet_placeholder = st.empty()
            code_header_placeholder.header(f"**Code for the step: 00 - Show  Dataset**")
            snippet_placeholder.code(snippet)

    elif nlp_steps == '02- Basic Information':
        st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
        st.text('* Dataset Size')
        st.write(df.shape)
        st.text('* Statistical Description of the Dataset')
        st.write(df.describe())
        st.markdown("---")

        st.markdown("We are generating for you a data quality report with the help of Pandas Profiling.")
        st.text('In function of the dataset size might take up to 1 min')
        pr = df.profile_report()
        st_profile_report(pr)
        st.markdown("---")

        snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np

        # Shape
        >>> df.shape

        # &

        # Statistical
        >>> df.describe()


        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 01 - Basic Information**")
        snippet_placeholder.code(snippet)

    elif nlp_steps == '03 - Exploratory Visualization':
        st.markdown("### i - Visualization fo the User ")
        chart_data = pd.DataFrame(
            np.random.randn(50, 3),
            columns=["a", "b", "c"])
        st.bar_chart(chart_data)

        st.markdown("### ii - Visualization of the betting process ")

        st.markdown("#### - Exploratory Dataset ")
        images2 = Image.open('images/Exploratory.png')
        st.image(images2, width=800)

        st.markdown("#### - Exploratory Winning Bid 1")
        images3 = Image.open('images/Exploratory2-winning bid.png')
        st.image(images3, width=800)

        st.markdown("#### - Exploratory Winning Bid 2")
        images4 = Image.open('images/segment_winning_bid.png')
        st.image(images4, width=800)

        st.markdown("#### - Exploratory Winning Bid 3")
        images5 = Image.open('images/winning3.png')
        st.image(images5, width=800)


        snippet = f"""

        >>> chart_data = pd.DataFrame(
        >>> np.random.randn(50, 3),
        >>> columns = ["a", "b", "c"])
        >>> st.bar_chart(chart_data)

        """
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        code_header_placeholder.header(f"**Code for the step: 02 - Visualization**")
        snippet_placeholder.code(snippet)

    elif nlp_steps == '04 - See information collected':
        st.markdown('In Progress')
        job_position = st.text_input("Enter your job position", "")
        age = st.slider("Pick your age", min_value=1, max_value=100, value=50, step=1)
        sex = st.multiselect("Select your gender",['Female','Male','Other'])
        is_often_online = st.checkbox("Do you often accept cookies")
        st.markdown(
            f"""
            * Name : {job_position}
            * Age : {age}
            * Sex : {sex}
            * Often Accepting Cookies : {is_often_online}
"""
        )

        if job_position != "":
            images6 = Image.open('images/brands.png')
            st.image(images6, width=800)

    elif nlp_steps == '05 - How your information is processed':

        st.markdown('In Progress')


if __name__ == '__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👨🏼‍🎓‍ Researcher and professor: **")
st.image(['images/klaus-miller.png'], width=100, caption=["Klaus Miller"])

st.markdown(
    f"####  Link to the Researcher Website [here]({'https://www.hec.edu/en/faculty-research/faculty-directory/faculty-member/miller-klausm.'}) 🚀 ")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer2():
    myargs = [
        " Made by ",
        link("https://engineeringteam.hi-paris.fr/", " Hi! PARIS "),
        "Engineering Team 👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()
