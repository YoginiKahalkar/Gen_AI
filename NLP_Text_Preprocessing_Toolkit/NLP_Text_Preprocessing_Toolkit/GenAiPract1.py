import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# --------------------------
# Load NLP Model
# --------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------
# Page Config
# --------------------------

st.set_page_config(
    page_title="NLP Text Preprocessing Toolkit",
    page_icon="🧠",
    layout="wide"
)


# --------------------------
# Custom CSS
# --------------------------

st.markdown(
"""
<style>

.stApp{
    background:linear-gradient(
        135deg,
        #667eea,
        #764ba2,
        #06beb6
    );
}


.block-container{

    background:rgba(255,255,255,0.85);

    padding:35px;

    border-radius:25px;

}



/* Headings */

h1,h2,h3,h4{

    color:#1f2937 !important;

    font-weight:700;

}



/* Normal Text */

p{

    color:#111827 !important;

}


label{

    color:#111827 !important;

    font-weight:600;

}



/* Text Area */

textarea{

    background:white !important;

    color:#111827 !important;

    border-radius:15px !important;

    font-size:16px !important;

}



/* Sidebar */

[data-testid="stSidebar"]{

    background:linear-gradient(
        180deg,
        #4f46e5,
        #7c3aed
    );

}



[data-testid="stSidebar"] *{

    color:white !important;

}



/* Button */

.stButton button{

    background:linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );


    color:white !important;

    font-size:18px;

    font-weight:bold;


    height:50px;

    width:220px;


    border-radius:30px;

    border:none;

}



.stButton button:hover{

    transform:scale(1.05);

}



/* Metrics */

[data-testid="metric-container"]{


    background:white;


    padding:20px;


    border-radius:20px;


    box-shadow:0px 5px 15px rgba(0,0,0,0.15);


}



[data-testid="metric-container"] label{

    color:#374151 !important;

}



[data-testid="metric-container"] div{

    color:#111827 !important;

}



/* Dataframes */

[data-testid="stDataFrame"]{

    background:white;

    border-radius:15px;

}



[data-testid="stDataFrame"] *{

    color:#111827 !important;

}


</style>

""",
unsafe_allow_html=True
)



# --------------------------
# Sidebar
# --------------------------

with st.sidebar:


    st.title("🧠 NLP Toolkit")


    st.success(
        "Developed by\n\nYogini Kahalkar"
    )


    st.markdown("---")


    st.subheader("Features")


    st.write("✅ Tokenization")

    st.write("✅ Stopword Removal")

    st.write("✅ POS Tagging")

    st.write("✅ Lemmatization")

    st.write("✅ Named Entity Recognition")

    st.write("✅ Dependency Parsing")

    st.write("✅ Sentence Segmentation")

    st.write("✅ Word Frequency")
    # --------------------------
# Title Section
# --------------------------

st.title(
    "🧠 NLP Text Preprocessing Toolkit"
)


st.markdown(
"""
<div style="
background:rgba(255,255,255,0.15);
padding:30px;
border-radius:25px;
text-align:center;
font-size:20px;
color:#111827;
">

✨ Explore Natural Language Processing techniques using

<br>

<b>spaCy</b> and <b>Streamlit</b>

<br><br>

Analyze text with AI-powered preprocessing.

</div>
""",
unsafe_allow_html=True
)



# --------------------------
# Input Box
# --------------------------

text = st.text_area(
    "✍ Enter your text here",
    height=200,
    placeholder="Example: Apple Inc. was founded by Steve Jobs in California in 1976."
)



# --------------------------
# Button Start
# --------------------------

if st.button("🚀 Analyze Text"):


    if text.strip()=="":
        st.warning("Please enter text")
        st.stop()



    doc = nlp(text)



    tokens=[

        token.text

        for token in doc

    ]



    filtered=[

        token.text

        for token in doc

        if not token.is_stop and token.is_alpha

    ]



    # --------------------------
    # Text Statistics
    # --------------------------

    sentences = list(doc.sents)

    entities = list(doc.ents)



    st.markdown("---")



    st.header("📊 Text Statistics")



    col1, col2, col3, col4, col5 = st.columns(5)



    col1.metric(
        "Characters",
        len(text)
    )


    col2.metric(
        "Words",
        len(tokens)
    )


    col3.metric(
        "Unique Words",
        len(set(filtered))
    )


    col4.metric(
        "Sentences",
        len(sentences)
    )


    col5.metric(
        "Entities",
        len(entities)
    )



    st.markdown("---")



    # --------------------------
    # Tokenization
    # --------------------------

    st.subheader(
        "1️⃣ Tokenization"
    )



    token_df = pd.DataFrame(
        {
            "Tokens": tokens
        }
    )



    st.dataframe(
        token_df,
        use_container_width=True
    )



    # --------------------------
    # Stopword Removal
    # --------------------------

    st.subheader(
        "2️⃣ Stopword Removal"
    )



    stop_df = pd.DataFrame(
        {
            "Filtered Words": filtered
        }
    )



    st.dataframe(
        stop_df,
        use_container_width=True
    )



    # --------------------------
    # POS Tagging
    # --------------------------

    st.subheader(
        "3️⃣ POS Tagging"
    )



    pos_list = []



    for token in doc:

        pos_list.append(

            {

                "Word": token.text,

                "POS": token.pos_,

                "Tag": token.tag_

            }

        )



    pos_df = pd.DataFrame(
        pos_list
    )



    st.dataframe(
        pos_df,
        use_container_width=True
    )
    # --------------------------
    # Lemmatization
    # --------------------------

    st.subheader(
        "4️⃣ Lemmatization"
    )



    lemma_list = []



    for token in doc:

        lemma_list.append(

            {
                "Word": token.text,

                "Lemma": token.lemma_,

                "POS": token.pos_

            }

        )



    lemma_df = pd.DataFrame(
        lemma_list
    )



    st.dataframe(
        lemma_df,
        use_container_width=True
    )



    # --------------------------
    # Named Entity Recognition
    # --------------------------

    st.subheader(
        "5️⃣ Named Entity Recognition (NER)"
    )



    entity_list = []



    for ent in doc.ents:

        entity_list.append(

            {
                "Entity": ent.text,

                "Label": ent.label_,

                "Description": spacy.explain(ent.label_)

            }

        )



    if entity_list:


        entity_df = pd.DataFrame(
            entity_list
        )


        st.dataframe(
            entity_df,
            use_container_width=True
        )


    else:

        st.info(
            "No entities found"
        )



    # --------------------------
    # Dependency Parsing
    # --------------------------

    st.subheader(
        "6️⃣ Dependency Parsing"
    )



    dependency_list = []



    for token in doc:


        dependency_list.append(

            {

                "Word": token.text,

                "Dependency": token.dep_,

                "Head": token.head.text,

                "Relation": spacy.explain(token.dep_)

            }

        )



    dependency_df = pd.DataFrame(
        dependency_list
    )



    st.dataframe(
        dependency_df,
        use_container_width=True
    )



    # --------------------------
    # Sentence Segmentation
    # --------------------------

    st.subheader(
        "7️⃣ Sentence Segmentation"
    )



    sentence_list = []



    for index, sentence in enumerate(sentences, 1):


        sentence_list.append(

            {

                "Sentence No": index,

                "Sentence": sentence.text

            }

        )



    sentence_df = pd.DataFrame(
        sentence_list
    )



    st.dataframe(
        sentence_df,
        use_container_width=True
    )
    # --------------------------
    # Token Details
    # --------------------------

    st.subheader(
        "8️⃣ Token Details"
    )



    token_details = []



    for token in doc:


        token_details.append(

            {

                "Token": token.text,

                "Lower": token.lower_,

                "Is Alphabet": token.is_alpha,

                "Is Stopword": token.is_stop,

                "Shape": token.shape_,

                "POS": token.pos_

            }

        )



    token_df = pd.DataFrame(
        token_details
    )



    st.dataframe(
        token_df,
        use_container_width=True
    )



    # --------------------------
    # Word Frequency
    # --------------------------

    st.subheader(
        "9️⃣ Word Frequency Analysis"
    )



    words = [

        token.text.lower()

        for token in doc

        if token.is_alpha

        and not token.is_stop

    ]



    frequency = Counter(words)



    freq_df = pd.DataFrame(

        frequency.items(),

        columns=[

            "Word",

            "Frequency"

        ]

    )



    freq_df = freq_df.sort_values(

        by="Frequency",

        ascending=False

    )



    st.dataframe(
        freq_df,
        use_container_width=True
    )



    # --------------------------
    # Chart
    # --------------------------

    st.subheader(
        "📈 Top Frequent Words"
    )



    if len(freq_df) > 0:


        top = freq_df.head(10)



        fig, ax = plt.subplots(

            figsize=(10,5)

        )



        ax.bar(

            top["Word"],

            top["Frequency"]

        )



        ax.set_xlabel(
            "Words"
        )


        ax.set_ylabel(
            "Frequency"
        )


        ax.set_title(
            "Top 10 Frequent Words"
        )



        plt.xticks(
            rotation=45
        )



        st.pyplot(fig)



# --------------------------
# Footer
# --------------------------

st.markdown(

"""
<div style="
background:rgba(255,255,255,0.15);
padding:25px;
border-radius:25px;
text-align:center;
color:#111827;
font-size:18px;
">

🧠 NLP Text Preprocessing Toolkit

<br><br>

Developed using

<br>

<b>
Python | Streamlit | spaCy
</b>

<br><br>

👤 Developed by

<b>
Yogini Kahalkar
</b>

</div>

""",

unsafe_allow_html=True

)