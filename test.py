import folium
import streamlit as st
from streamlit_folium import st_folium

form = st.form("location_form")
set_latitude = form.number_input("Latitude")
set_longitude = form.number_input("Longitude")
submit_button = form.form_submit_button("Submit")

# Initialize markers inside of session state
if "markers" not in st.session_state:
    st.session_state["markers"] = []

location = [set_latitude, set_longitude]
m = folium.Map(location=location, zoom_start=16)

if submit_button:
    st.session_state["markers"].append(folium.Marker(location=location))
    st.write(location)

fg = folium.FeatureGroup(name="Markers")
for marker in st.session_state["markers"]:
    fg.add_child(marker)
st_data = st_folium(m, feature_group_to_add=fg, width=640, height=320)
st.write(st_data)