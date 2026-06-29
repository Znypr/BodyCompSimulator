import streamlit as st
import logic


def render_training_inputs():
    with st.expander("Training and nutrition", expanded=True):
        status = st.selectbox(
            "Resistance-training status",
            list(logic.MAX_FFM_GAIN_FRACTION_PER_MONTH),
            index=2,
        )
        quality = st.slider(
            "Training quality and consistency",
            0.0,
            1.0,
            0.9,
            0.05,
        )
        protein = st.slider("Protein (g/kg/day)", 0.6, 3.0, 2.2, 0.1)
    return status, quality, protein
